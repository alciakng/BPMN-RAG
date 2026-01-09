# knowledge_augmenter.py
"""
ReAct Framework Step 2: External Knowledge Augmentation
외부 지식 저장소를 활용하여 개선 제안을 위한 인사이트 수집
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from common.logger import Logger
from manager.external_knowledge_store import ExternalKnowledgeStore

LOGGER = Logger.get_logger("agent.knowledge_augmenter")


class KnowledgeAugmenter:
    """
    External knowledge retrieval and organization for 2-Stage Agent

    Responsibilities:
    - Retrieve external knowledge based on intent analysis results
    - Organize knowledge by aspects (alternatives, best_practices, risks, etc.)
    - Structure insights for LLM consumption
    - Handle graceful degradation when external sources fail
    """

    def __init__(self, knowledge_store: ExternalKnowledgeStore):
        """
        Args:
            knowledge_store: External knowledge source (e.g., WebSearchKnowledgeStore)
        """
        if not knowledge_store:
            raise ValueError("Knowledge store is required for KnowledgeAugmenter")

        self.store = knowledge_store

    def augment(
        self,
        user_query: str,
        stage1_answer: str,
        intent_result: Dict[str, Any],
        process_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Retrieve and organize external knowledge based on intent analysis

        Args:
            user_query: Original user query
            stage1_answer: Stage-1 answer from GraphDB
            intent_result: Intent analysis result from IntentAnalyzer
                {
                    "needs_insight": bool,
                    "reasoning": str,
                    "confidence": float
                }
            process_context: Process context
                {
                    "process_name": str,
                    "domain": str,
                    "selected_models": List[str],
                    "model_summaries": List[str]
                }

        Returns:
            {
                "has_insights": bool,                    # External knowledge found?
                "insights_by_aspect": Dict[str, List],   # {"external_knowledge": [...]}
                "total_sources": int,                    # Total knowledge items
                "context_summary": str,                  # Brief summary for LLM
                "metadata": {
                    "search_performed": bool,
                    "confidence": float
                }
            }
        """
        try:
            LOGGER.info("[AUGMENT] Starting knowledge augmentation")
            LOGGER.info("[AUGMENT] needs_insight=%s",
                       intent_result["needs_insight"])

            # Check if insight generation is needed
            if not intent_result.get("needs_insight"):
                LOGGER.info("[AUGMENT] No insight needed, skipping augmentation")
                return self._empty_result(reason="Intent analysis indicates no augmentation needed")

            # Search using user_query directly (no category-based enhancement)
            LOGGER.info("[AUGMENT] Searching with user query (no enhancement)")

            try:
                knowledge_items = self._search_external_knowledge(
                    user_query=user_query,
                    stage1_answer=stage1_answer,
                    process_context=process_context
                )

                total_sources = len(knowledge_items)

                if knowledge_items:
                    LOGGER.info("[AUGMENT] Found %d items", total_sources)
                else:
                    LOGGER.warning("[AUGMENT] No items found")

            except Exception as e:
                LOGGER.exception("[AUGMENT][ERROR] Failed to search: %s", e)
                knowledge_items = []
                total_sources = 0

            # Generate context summary
            context_summary = self._generate_context_summary(
                knowledge_items=knowledge_items
            )

            result = {
                "has_insights": total_sources > 0,
                "insights_by_aspect": {"external_knowledge": knowledge_items} if knowledge_items else {},
                "total_sources": total_sources,
                "context_summary": context_summary,
                "metadata": {
                    "search_performed": True,
                    "confidence": intent_result.get("confidence", 0.0)
                }
            }

            LOGGER.info("[AUGMENT] Complete - total_sources=%d", total_sources)

            return result

        except Exception as e:
            LOGGER.exception("[AUGMENT][ERROR] Knowledge augmentation failed: %s", e)
            return self._empty_result(reason=f"Augmentation error: {str(e)}")

    # ============================================================
    # External Knowledge Retrieval
    # ============================================================

    def _search_external_knowledge(
        self,
        user_query: str,
        stage1_answer: str,
        process_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search external knowledge using user query as-is

        Args:
            user_query: Original user query (used directly without enhancement)
            stage1_answer: Stage-1 answer
            process_context: Process context

        Returns:
            List of knowledge items with content, source, URL, relevance score
        """
        try:
            # Build search context
            search_context = {
                "process_name": process_context.get("process_name", "unknown"),
                "domain": process_context.get("domain", "general"),
                "current_issues": self._extract_issues_from_answer(stage1_answer),
                "selected_models": process_context.get("selected_models", [])
            }

            LOGGER.info("[SEARCH] query=%s", user_query[:100])

            # Call external knowledge store with user_query as-is
            # Trust Tavily API to understand the query without keyword enhancement
            knowledge_items = self.store.search(
                query=user_query,  # Use user query directly
                category="general",  # Single generic category
                context=search_context,
                top_k=5  # Broad (2) + Focused (3) = 5 total
            )

            LOGGER.info("[SEARCH] Retrieved %d items", len(knowledge_items))

            # Log content preview for each item
            for idx, item in enumerate(knowledge_items, 1):
                content_preview = item.get("content", "")[:50].replace("\n", " ")
                tier = item.get("tier", "unknown")
                source = item.get("source", "Unknown")
                LOGGER.info("[SEARCH] Item %d [%s] %s: %s...",
                           idx, tier.upper(), source[:30], content_preview)

            return knowledge_items

        except Exception as e:
            LOGGER.exception("[SEARCH][ERROR] error=%s", e)
            return []

    def _extract_issues_from_answer(self, stage1_answer: str) -> List[str]:
        """
        Extract current process issues mentioned in Stage-1 answer

        Uses simple heuristics to identify potential issues:
        - "병목" (bottleneck)
        - "지연" (delay)
        - "비효율" (inefficiency)
        - "중복" (duplication)
        - "리스크" (risk)

        Returns:
            List of identified issues (Korean)
        """
        issues = []

        # Simple keyword matching (can be enhanced with NER later)
        issue_keywords = {
            "병목": "bottleneck",
            "지연": "delay",
            "비효율": "inefficiency",
            "중복": "duplication",
            "리스크": "risk",
            "문제": "problem",
            "개선": "improvement needed"
        }

        answer_lower = stage1_answer.lower()
        for korean, english in issue_keywords.items():
            if korean in answer_lower:
                issues.append(korean)

        return issues

    # ============================================================
    # Context Summary Generation
    # ============================================================

    def _generate_context_summary(
        self,
        knowledge_items: List[Dict[str, Any]]
    ) -> str:
        """
        Generate concise summary of retrieved external knowledge

        This summary is used as context for AnswerAggregator LLM call

        Args:
            knowledge_items: Retrieved knowledge items

        Returns:
            Korean text summary (2-3 sentences)
        """
        if not knowledge_items:
            return "외부 지식 검색 결과가 없습니다."

        total_sources = len(knowledge_items)

        summary = f"질의와 관련된 외부 지식 {total_sources}건을 수집했습니다. "
        summary += "클라우드 아키텍처, 모범 사례, 업계 표준 등을 기반으로 구체적인 인사이트를 제공합니다."

        LOGGER.info("[CONTEXT_SUMMARY] Generated summary (len=%d): %s", len(summary), summary[:100])

        return summary

    # ============================================================
    # Utilities
    # ============================================================

    def _empty_result(self, reason: str = "") -> Dict[str, Any]:
        """
        Return empty augmentation result (graceful degradation)
        """
        LOGGER.info("[AUGMENT] Returning empty result: %s", reason)

        return {
            "has_insights": False,
            "insights_by_aspect": {},
            "total_sources": 0,
            "context_summary": "외부 지식 검색이 수행되지 않았습니다.",
            "metadata": {
                "search_performed": False,
                "reason": reason
            }
        }

    def format_for_llm(self, augmentation_result: Dict[str, Any]) -> str:
        """
        Format augmentation result for LLM consumption

        Returns:
            Structured text format for LLM prompt
        """
        if not augmentation_result.get("has_insights"):
            return "## 외부 지식 검색 결과\n검색된 외부 지식이 없습니다.\n"

        insights_by_aspect = augmentation_result["insights_by_aspect"]

        formatted = "## 외부 지식 검색 결과\n\n"
        formatted += f"**총 {augmentation_result['total_sources']}건의 신뢰할 수 있는 출처**\n\n"

        # Flatten all items (no category separation needed)
        all_items = []
        for items in insights_by_aspect.values():
            all_items.extend(items)

        for idx, item in enumerate(all_items, 1):
            source = item.get("source", "Unknown")
            url = item.get("url", "")
            content = item.get("content", "")
            tier = item.get("tier", "unknown")
            domain_source = item.get("metadata", {}).get("domain_source", "")

            # Tier indicator
            tier_label = "🌐" if tier == "broad" else "🎯"

            formatted += f"{tier_label} **{idx}. {source}**"

            if domain_source:
                formatted += f" (출처: {domain_source.upper()})"

            formatted += f"\n{content}\n"

            if url:
                formatted += f"[상세보기]({url})\n"

            formatted += "\n"

        return formatted
