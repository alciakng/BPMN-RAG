# answer_aggregator.py
"""
ReAct Framework Step 3: Answer Aggregation
Stage-1 답변과 외부 지식을 결합하여 섹션 분리된 최종 답변 생성
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from common.logger import Logger
from common.llm_client import LLMClient
from manager.util import _extract_text

LOGGER = Logger.get_logger("agent.answer_aggregator")


class AnswerAggregator:
    """
    Aggregate Stage-1 answer with external knowledge insights

    Responsibilities:
    - Combine GraphDB-based answer (Stage-1) with external knowledge (Stage-2)
    - Generate section-separated final answer using LLM
    - Ensure clear attribution and source citation
    - Handle cases where no external knowledge is available
    """

    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: LLM client for answer aggregation
        """
        if not llm_client:
            raise ValueError("LLM client is required for AnswerAggregator")

        self.llm = llm_client

    def aggregate(
        self,
        user_query: str,
        stage1_answer: str,
        augmentation_result: Dict[str, Any],
        intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate Stage-1 answer with external knowledge

        Args:
            user_query: Original user query
            stage1_answer: Stage-1 answer from GraphDB
            augmentation_result: External knowledge from KnowledgeAugmenter
            intent_result: Intent analysis result

        Returns:
            {
                "final_answer": str,              # Complete answer (Korean)
                "has_stage2": bool,               # Stage-2 insights included?
                "sections": {
                    "stage1": str,                # GraphDB-based answer
                    "stage2": Optional[str]       # External knowledge insights
                },
                "sources": List[Dict],            # External sources cited
                "metadata": {
                    "total_sources": int,
                    "generation_method": str      # "stage1_only" | "2stage_aggregated"
                }
            }
        """
        try:
            LOGGER.info("[AGGREGATE] Starting answer aggregation")
            LOGGER.info("[AGGREGATE] has_insights=%s, total_sources=%d",
                       augmentation_result.get("has_insights"),
                       augmentation_result.get("total_sources", 0))

            # Case 1: No external knowledge - return Stage-1 only
            if not augmentation_result.get("has_insights"):
                LOGGER.info("[AGGREGATE] No external insights, returning Stage-1 only")
                return self._stage1_only_result(
                    user_query=user_query,
                    stage1_answer=stage1_answer,
                    reason="No external knowledge found or not needed"
                )

            # Case 2: Aggregate Stage-1 + Stage-2
            LOGGER.info("[AGGREGATE] Aggregating Stage-1 + Stage-2")
            final_answer = self._llm_aggregate(
                user_query=user_query,
                stage1_answer=stage1_answer,
                augmentation_result=augmentation_result,
                intent_result=intent_result
            )

            if not final_answer:
                # Fallback if LLM aggregation fails
                LOGGER.warning("[AGGREGATE] LLM aggregation failed, using fallback format")
                final_answer = self._fallback_aggregate(
                    stage1_answer=stage1_answer,
                    augmentation_result=augmentation_result
                )

            # Extract sources
            sources = self._extract_sources(augmentation_result)

            result = {
                "final_answer": final_answer,
                "has_stage2": True,
                "sections": {
                    "stage1": stage1_answer,
                    "stage2": augmentation_result.get("context_summary", "")
                },
                "sources": sources,
                "metadata": {
                    "total_sources": augmentation_result.get("total_sources", 0),
                    "generation_method": "2stage_aggregated"
                }
            }

            LOGGER.info("[AGGREGATE] Complete - final_answer_len=%d, sources=%d",
                       len(final_answer), len(sources))

            return result

        except Exception as e:
            LOGGER.exception("[AGGREGATE][ERROR] Aggregation failed: %s", e)
            # Fallback to Stage-1 only
            return self._stage1_only_result(
                user_query=user_query,
                stage1_answer=stage1_answer,
                reason=f"Aggregation error: {str(e)}"
            )

    # ============================================================
    # LLM-based Aggregation
    # ============================================================

    def _llm_aggregate(
        self,
        user_query: str,
        stage1_answer: str,
        augmentation_result: Dict[str, Any],
        intent_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Use LLM to generate section-separated final answer

        Returns:
            Final answer in Korean with clear section separation
        """
        try:
            # Build external knowledge context text
            external_knowledge_text = self._format_external_knowledge(augmentation_result)

            system_prompt = """
[ROLE]
You are a BPMN Process Intelligence expert providing comprehensive answers.

[TASK]
Combine the GraphDB-based answer (Stage-1) with external knowledge insights (Stage-2) to provide a complete, actionable answer.

[OUTPUT FORMAT]
Generate a well-structured Korean answer with TWO clear sections:

---
##  프로세스 분석 결과

[Stage-1 answer content - preserve original format and structure]
- DO NOT summarize or paraphrase the Stage-1 answer
- Keep all tables, bullet points, and formatting as-is
- Only compress if the content is excessively long (>2000 characters):
  - Remove redundant explanations
  - Condense verbose descriptions while keeping key points
  - Maintain all critical data (IDs, metrics, evidence)

---
## 개선 제안 및 참고 사항

[Synthesize external knowledge into a consulting-style insight report]

### Executive Summary
- **핵심 발견사항**: [1-2문장으로 외부 지식에서 발견된 핵심 인사이트 요약]
- **비즈니스 임팩트**: [현재 프로세스 대비 개선 시 기대효과를 정량적/정성적으로 제시]

### 전략적 개선 방향

#### 1. [개선 영역 1 - 예: 프로세스 자동화 확대]
- **현황**: [현재 GraphDB 분석 결과와 연계한 문제점]
- **Best Practice**: [외부 지식 기반 모범 사례 - 구체적 사례/수치 포함]
- **권장 사항**: [실행 가능한 3-5개의 구체적 액션 아이템]
- **구현 난이도**: [상/중/하] | **예상 효과**: [높음/중간/낮음]

#### 2. [개선 영역 2 - 예: 리스크 관리 강화]
- **현황**: [현재 GraphDB 분석 결과와 연계한 문제점]
- **Best Practice**: [외부 지식 기반 모범 사례]
- **권장 사항**: [실행 가능한 액션 아이템]
- **구현 난이도**: [상/중/하] | **예상 효과**: [높음/중간/낮음]

[필요시 개선 영역 3, 4 추가...]

### 리스크 및 고려사항
- [대안 도입 시 발생 가능한 리스크 1 - 완화 방안 포함]
- [대안 도입 시 발생 가능한 리스크 2 - 완화 방안 포함]
- [조직/기술적 제약사항 - 실행 시 주의사항]

### 벤치마크 및 산업 표준
- [관련 산업/도메인의 표준 프로세스 레퍼런스 - APQC, ISO 등]
- [주요 기업 사례 - AWS/Azure/GCP Well-Architected Framework 기반]
- [정량적 벤치마크 수치 - 가능한 경우]

### 🔗 참고 자료 및 출처
1. **[Source 1 Title]** - [핵심 내용 1-2문장 요약]
   🔗 [URL]
2. **[Source 2 Title]** - [핵심 내용 1-2문장 요약]
   🔗 [URL]

---

[REQUIREMENTS]
1. **Preserve Stage-1 Format**: Keep original markdown structure (tables, bullets, code blocks, bold text)
2. **Section Separation**: Clearly separate Stage-1 and Stage-2 with markdown headers
3. **Consulting-Style Insights**: Write "개선 제안 및 참고 사항" as a professional consulting report with:
   - Executive Summary (핵심 발견사항 + 비즈니스 임팩트)
   - Strategic Improvement Areas (전략적 개선 방향) - organized by themes
   - Risk Assessment (리스크 및 고려사항)
   - Benchmarks & Industry Standards (벤치마크 및 산업 표준)
4. **Evidence-Based**: Link Stage-1 findings to external knowledge insights
5. **Quantitative When Possible**: Include metrics, ROI estimates, benchmark numbers
6. **Actionable & Prioritized**: Provide specific action items with difficulty/impact ratings
7. **Source Attribution**: Cite external sources with brief summaries and URLs
8. **Korean**: All content must be in Korean
9. **Synthesis Over Listing**: Integrate external knowledge into coherent strategic narrative

[IMPORTANT]
- DO NOT rewrite or summarize Stage-1 unless it exceeds 2000 characters
- **Connect Stage-1 to Stage-2**: Explicitly link GraphDB findings to external insights in "현황" field
- **Be Specific**: Replace generic advice with concrete examples from sources (e.g., "AWS recommends X pattern for Y scenario")
- **Assess Trade-offs**: For each recommendation, mention implementation complexity and expected impact
- If external knowledge contradicts Stage-1, present both perspectives and recommend reconciliation approach
- Prioritize tier-1 enterprise sources (AWS, Azure, GCP, APQC) over generic content
- Structure improvements by strategic themes (automation, risk management, cost optimization, etc.)
- Include industry benchmarks when available to contextualize current performance
"""

            user_message = f"""
[USER QUERY]
{user_query}

[STAGE-1 ANSWER]
{stage1_answer}

{external_knowledge_text}

[INTENT ANALYSIS]
- Needs Insight: {intent_result.get('needs_insight', False)}
- Confidence: {intent_result.get('confidence', 0.0):.2f}

[TASK]
Generate the final aggregated answer by combining Stage-1 answer with external knowledge insights.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            LOGGER.info("[AGGREGATE][LLM] Calling LLM for aggregation")
            raw_response = self.llm.complete(messages)
            final_answer = _extract_text(raw_response)

            LOGGER.info("[AGGREGATE][LLM] Generated answer, length=%d", len(final_answer))

            return final_answer

        except Exception as e:
            LOGGER.exception("[AGGREGATE][LLM][ERROR] %s", e)
            return None

    # ============================================================
    # Fallback Aggregation (Template-based)
    # ============================================================

    def _fallback_aggregate(
        self,
        stage1_answer: str,
        augmentation_result: Dict[str, Any]
    ) -> str:
        """
        Template-based aggregation when LLM fails

        Returns:
            Simple concatenation with section headers
        """
        LOGGER.info("[AGGREGATE][FALLBACK] Using template-based aggregation")

        final_answer = "---\n## 📊 프로세스 분석 결과 (GraphDB 기반)\n\n"
        final_answer += stage1_answer
        final_answer += "\n\n---\n## 💡 개선 제안 및 참고 사항 (외부 지식 기반)\n\n"

        insights_by_aspect = augmentation_result.get("insights_by_aspect", {})

        aspect_korean = {
            "background": "배경 지식",
            "alternatives": "대안 설계",
            "best_practices": "모범 사례",
            "risks": "리스크 분석",
            "quantitative": "정량적 지표"
        }

        if insights_by_aspect:
            for aspect, items in insights_by_aspect.items():
                aspect_name = aspect_korean.get(aspect, aspect)
                final_answer += f"### {aspect_name}\n\n"

                for idx, item in enumerate(items, 1):
                    content = item.get("content", "")
                    source = item.get("source", "Unknown")
                    url = item.get("url", "")

                    final_answer += f"**{idx}. {source}**\n"
                    final_answer += f"{content}\n"

                    if url:
                        final_answer += f"[상세보기]({url})\n"

                    final_answer += "\n"
        else:
            final_answer += "외부 지식 검색 결과가 없습니다.\n"

        final_answer += "\n---\n"

        return final_answer

    # ============================================================
    # Stage-1 Only Result
    # ============================================================

    def _stage1_only_result(
        self,
        user_query: str,
        stage1_answer: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Return Stage-1 only result (no external knowledge)
        """
        LOGGER.info("[AGGREGATE] Returning Stage-1 only result: %s", reason)

        return {
            "final_answer": stage1_answer,
            "has_stage2": False,
            "sections": {
                "stage1": stage1_answer,
                "stage2": None
            },
            "sources": [],
            "metadata": {
                "total_sources": 0,
                "generation_method": "stage1_only",
                "reason": reason
            }
        }

    # ============================================================
    # Utilities
    # ============================================================

    def _format_external_knowledge(self, augmentation_result: Dict[str, Any]) -> str:
        """
        Format augmentation result as text for LLM consumption

        Returns:
            Formatted text describing external knowledge by aspect
        """
        insights_by_aspect = augmentation_result.get("insights_by_aspect", {})

        if not insights_by_aspect:
            return "[EXTERNAL KNOWLEDGE]\nNo external knowledge found."

        lines = ["[EXTERNAL KNOWLEDGE]"]

        for category, items in insights_by_aspect.items():
            lines.append(f"\n## External Knowledge\n")

            for idx, item in enumerate(items, 1):
                content = item.get("content", "")
                source = item.get("source", "Unknown")
                url = item.get("url", "")
                tier = item.get("tier", "unknown")

                lines.append(f"{idx}. [{tier.upper()}] {source}")
                lines.append(f"   {content[:200]}...")
                if url:
                    lines.append(f"   URL: {url}")
                lines.append("")

        return "\n".join(lines)

    def _extract_sources(self, augmentation_result: Dict[str, Any]) -> list[Dict[str, str]]:
        """
        Extract unique sources from augmentation result

        Returns:
            [
                {
                    "title": str,
                    "url": str,
                    "domain": str,  # "aws" | "azure" | "gcp" | "apqc"
                    "tier": str     # "broad" | "focused"
                }
            ]
        """
        sources = []
        seen_urls = set()

        insights_by_aspect = augmentation_result.get("insights_by_aspect", {})

        for category, items in insights_by_aspect.items():
            for item in items:
                url = item.get("url", "")

                # Skip duplicates
                if url in seen_urls:
                    continue

                seen_urls.add(url)

                sources.append({
                    "title": item.get("source", "Unknown"),
                    "url": url,
                    "domain": item.get("metadata", {}).get("domain_source", "unknown"),
                    "tier": item.get("tier", "unknown"),
                    "category": category
                })

        LOGGER.info("[EXTRACT_SOURCES] Extracted %d unique sources", len(sources))
        return sources
