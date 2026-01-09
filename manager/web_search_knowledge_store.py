# web_search_knowledge_store.py
"""
Tavily API-based Web Search Knowledge Store
2-tier search strategy: Broad + Focused (Allowlist)
"""
from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from manager.external_knowledge_store import ExternalKnowledgeStore
from common.logger import Logger
from common.llm_client import LLMClient
from manager.util import _extract_text

LOGGER = Logger.get_logger("manager.web_search_knowledge_store")


class WebSearchKnowledgeStore(ExternalKnowledgeStore):
    """
    Tavily API-based two-tier knowledge search

    Architecture:
        Tier 1 (Broad): General background knowledge (top 3)
        Tier 2 (Focused): Domain-specific best practices from trusted sources (top 5-7)

    Allowlist domains:
        - Process standards: apqc.org
        - AWS: aws.amazon.com/solutions, docs.aws.amazon.com/prescriptive-guidance
        - Azure: learn.microsoft.com/azure/architecture, techcommunity.microsoft.com
        - GCP: cloud.google.com/architecture, cloud.google.com/docs
    """

    # Tier 2 allowlist (trusted enterprise architecture sources)
    ALLOWLIST_DOMAINS = [
        # Process standards
        "apqc.org",

        # AWS
        "aws.amazon.com/solutions",
        "docs.aws.amazon.com/prescriptive-guidance",

        # Azure
        "learn.microsoft.com/azure/architecture",
        "techcommunity.microsoft.com",

        # Google Cloud
        "cloud.google.com/architecture",
        "cloud.google.com/docs"
    ]

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Args:
            tavily_api_key: Tavily API key (or set TAVILY_API_KEY env var)
            llm_client: LLM client for content refinement (optional)
        """
        self.api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not provided. Set env var or pass as argument.")

        self.llm = llm_client
        self.tavily = TavilyClient(api_key=self.api_key)

        # Simple in-memory cache (query hash -> results)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

        LOGGER.info("[WEBSEARCH] Initialized with Tavily API")

    def search(
        self,
        query: str,
        category: str,
        context: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Two-tier search strategy combining broad background and focused domain searches

        Args:
            query: Search query (category-specific from KnowledgeAugmenter)
            category: Intent category (improvement, comparison, risk_analysis, background)
            context: Process context
            top_k: Maximum results to return

        Returns:
            Combined results from Tier 1 (broad) + Tier 2 (focused)
        """
        try:
            LOGGER.info("[WEBSEARCH] Start search category=%s top_k=%d", category, top_k)

            # Check cache
            cache_key = self._make_cache_key(query, category, context)
            if cache_key in self._cache:
                LOGGER.info("[WEBSEARCH][CACHE] Hit for query: %s", query[:50])
                return self._cache[cache_key]

            # Tier 1: Broad background search (2 results)
            broad_results = self._search_broad_background(
                query=query,
                context=context,
                top_k=2
            )
            LOGGER.info("[WEBSEARCH] Tier 1 (broad): %d results", len(broad_results))

            # Tier 2: Focused domain search (3 results)
            focused_results = self._search_focused_domain(
                query=query,
                category=category,
                context=context,
                top_k=3
            )
            LOGGER.info("[WEBSEARCH] Tier 2 (focused): %d results", len(focused_results))

            # Merge and limit to top_k
            all_results = broad_results + focused_results
            final_results = all_results[:top_k]

            # Store in cache
            self._cache[cache_key] = final_results

            LOGGER.info("[WEBSEARCH] Complete, returning %d items", len(final_results))
            return final_results

        except Exception as e:
            LOGGER.exception("[WEBSEARCH][ERROR] %s", e)
            return []

    # ============================================================
    # Tier 1: Broad Background Search
    # ============================================================

    def _search_broad_background(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tier 1: Broad search for general background knowledge

        - search_depth: "basic" (faster, broader coverage)
        - No domain restrictions
        - Returns top 3 results for context
        """
        try:
            LOGGER.info("[BROAD] Starting background search, top_k=%d", top_k)

            # Use query as-is (query is already category-optimized from KnowledgeAugmenter)
            search_query = query
            LOGGER.info("[BROAD] Query: %s", search_query)

            # Tavily search with basic depth
            response = self.tavily.search(
                query=search_query,
                search_depth="basic",          # Fast, broad coverage
                max_results=top_k * 2,         # Get more for filtering
                include_answer=False,          # We'll extract ourselves
                include_raw_content=True,      # Get full content
                # NO include_domains restriction for broad search
            )

            # Process results
            knowledge_items = []
            for idx, result in enumerate(response.get("results", [])[:top_k]):

                # Safe extraction - ensure result is a dict
                if not isinstance(result, dict):
                    LOGGER.warning("[BROAD] Skipping non-dict result at position %d", idx + 1)
                    continue

                # Safe extraction of raw_content length
                raw_content = result.get("raw_content") or ""
                raw_content_len = len(raw_content) if isinstance(raw_content, str) else 0

                item = {
                    "content": result.get("content", ""),
                    "source": result.get("title", "Unknown"),
                    "url": result.get("url", ""),
                    "relevance_score": result.get("score", 0.5),
                    "tier": "broad",
                    "metadata": {
                        "category": "background",
                        "published_date": result.get("published_date", ""),
                        "raw_content_length": raw_content_len,
                        "position": idx + 1
                    }
                }

                # Optionally: LLM-based refinement for better extraction
                if self.llm and result.get("raw_content"):
                    refined = self._llm_refine_background(
                        raw_content=result["raw_content"][:2000],  # Limit length
                        context=context
                    )
                    if refined:
                        item["content"] = refined
                        item["metadata"]["llm_refined"] = True

                knowledge_items.append(item)
                LOGGER.info(
                    "[BROAD] Added item %d: %s (score=%.3f)",
                    idx + 1,
                    item["source"][:50],
                    item["relevance_score"]
                )

            LOGGER.info("[BROAD] Completed, found %d items", len(knowledge_items))
            return knowledge_items

        except Exception as e:
            LOGGER.exception("[BROAD][ERROR] %s", e)
            return []

    # def _build_broad_query(self, query: str, context: Dict[str, Any]) -> str:
    #     """
    #     Build optimized query for broad background search
    #
    #     DEPRECATED: Query is now used as-is from KnowledgeAugmenter._build_category_query
    #     which already includes category-specific optimization.
    #
    #     Args:
    #         query: Category-specific query from KnowledgeAugmenter
    #         context: Process context
    #
    #     Returns:
    #         Broad search query
    #     """
    #     # No longer needed - query is already optimized by KnowledgeAugmenter
    #     return query.strip()

    def _llm_refine_background(
        self,
        raw_content: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Use LLM to extract concise background knowledge from raw content

        Returns:
            Refined 2-3 sentence summary (Korean), or None if irrelevant
        """
        if not self.llm:
            return None

        try:
            prompt = f"""
[TASK]
Extract background knowledge relevant to the process below.

[PROCESS CONTEXT]
- Process: {context.get("process_name", "unknown")}
- Domain: {context.get("domain", "unknown")}

[WEB CONTENT]
{raw_content[:2000]}...

[EXTRACTION RULES]
1. Extract ONLY background/context information (standards, regulations, industry trends)
2. Focus on industry practices, compliance requirements, domain knowledge
3. Write in Korean, 2-3 sentences, concise
4. Include specific standards/sources if mentioned (e.g., "ISO 9001에 따르면...")

[OUTPUT]
Return plain text (no JSON, no markdown).
If content is irrelevant to the process context, return: IRRELEVANT
"""

            messages = [
                {"role": "system", "content": "You are a background knowledge extraction expert."},
                {"role": "user", "content": prompt}
            ]

            response = self.llm.complete(messages)
            text = _extract_text(response).strip()

            if "IRRELEVANT" in text.upper() or len(text) < 20:
                return None

            return text

        except Exception as e:
            LOGGER.warning("[LLM_REFINE][ERROR] %s", e)
            return None

    # ============================================================
    # Tier 2: Focused Domain Search (Allowlist)
    # ============================================================

    def _search_focused_domain(
        self,
        query: str,
        category: str,
        context: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Tier 2: Focused search with allowlist domains

        - search_depth: "advanced" (comprehensive)
        - include_domains: ALLOWLIST_DOMAINS
        - Returns top 5 results for improvement recommendations

        Args:
            query: Search query from KnowledgeAugmenter
            category: Intent category (improvement, comparison, risk_analysis, background)
            context: Process context
            top_k: Maximum results to return
        """
        try:
            LOGGER.info("[FOCUSED] Starting domain search, category=%s, top_k=%d", category, top_k)

            # Use query as-is (query is already category-optimized from KnowledgeAugmenter)
            search_query = query
            LOGGER.info("[FOCUSED] Query: %s", search_query)

            # Tavily search with advanced depth + allowlist
            response = self.tavily.search(
                query=search_query,
                search_depth="advanced",               # Deep, comprehensive search
                max_results=top_k * 2,
                include_answer=False,
                include_raw_content=True,
                include_domains=self.ALLOWLIST_DOMAINS  # ✅ Allowlist restriction
            )

            # Process results
            knowledge_items = []
            for idx, result in enumerate(response.get("results", [])[:top_k]):
                # Safe extraction - ensure result is a dict
                if not isinstance(result, dict):
                    LOGGER.warning("[FOCUSED] Skipping non-dict result at position %d", idx + 1)
                    continue

                # Identify domain source
                url = result.get("url", "")
                domain_source = self._identify_domain_source(url)

                # Extract structured insights using LLM (if available)
                if self.llm:
                    # Safe extraction
                    raw_content = result.get("raw_content") or result.get("content") or ""
                    if isinstance(raw_content, str):
                        raw_content = raw_content[:3000]
                    else:
                        LOGGER.warning("[FOCUSED] raw_content is not str: type=%s", type(raw_content))
                        raw_content = ""

                    insights = self._llm_extract_insights(
                        raw_content=raw_content,
                        category=category,
                        context=context,
                        domain_source=domain_source
                    )
                else:
                    # Fallback: use Tavily content as-is
                    insights = {
                        "content": result.get("content", ""),
                        "pattern": "",
                        "complexity": "Medium",
                        "citations": []
                    }

                if not insights or not insights.get("content"):
                    continue

                item = {
                    "content": insights["content"],
                    "source": result.get("title", "Unknown"),
                    "url": result.get("url", ""),
                    "relevance_score": result.get("score", 0.5),
                    "tier": "focused",
                    "metadata": {
                        "category": category,
                        "domain_source": domain_source,
                        "architecture_pattern": insights.get("pattern", ""),
                        "implementation_complexity": insights.get("complexity", "Medium"),
                        "citations": insights.get("citations", []),
                        "position": idx + 1
                    }
                }

                knowledge_items.append(item)
                LOGGER.info(
                    "[FOCUSED] Added item %d: %s from %s (score=%.3f)",
                    idx + 1,
                    item["source"][:40],
                    domain_source,
                    item["relevance_score"]
                )

            LOGGER.info("[FOCUSED] Completed, found %d items", len(knowledge_items))
            return knowledge_items

        except Exception as e:
            LOGGER.exception("[FOCUSED][ERROR] %s", e)
            return []

    # def _build_focused_query(
    #     self,
    #     query: str,
    #     category: str,
    #     context: Dict[str, Any]
    # ) -> str:
    #     """
    #     Build optimized query for focused domain search
    #
    #     DEPRECATED: Query is now used as-is from KnowledgeAugmenter._build_category_query
    #     which already includes category-specific optimization.
    #
    #     Args:
    #         query: Category-specific query from KnowledgeAugmenter
    #         category: Intent category (improvement, comparison, risk_analysis, background)
    #         context: Process context
    #
    #     Returns:
    #         Focused search query
    #     """
    #     # No longer needed - query is already optimized by KnowledgeAugmenter
    #     return query.strip()

    def _identify_domain_source(self, url: str) -> str:
        """
        Identify which allowlist domain the URL belongs to

        Returns:
            "apqc" | "aws" | "azure" | "gcp" | "unknown"
        """
        url_lower = url.lower()

        if "apqc.org" in url_lower:
            return "apqc"
        elif "aws.amazon.com" in url_lower or "docs.aws.amazon.com" in url_lower:
            return "aws"
        elif "microsoft.com" in url_lower or "azure" in url_lower:
            return "azure"
        elif "cloud.google.com" in url_lower or "google.com" in url_lower:
            return "gcp"
        else:
            return "unknown"

    def _llm_extract_insights(
        self,
        raw_content: str,
        category: str,
        context: Dict[str, Any],
        domain_source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured insights from domain-specific content using LLM

        Args:
            raw_content: Raw content from Tavily search result
            category: Intent category (improvement, comparison, risk_analysis, background)
            context: Process context
            domain_source: Domain source (apqc, aws, azure, gcp)

        Returns:
            {
                "content": str,              # 2-3 sentences in Korean
                "pattern": str,              # Architecture pattern name
                "complexity": "Low|Medium|High",
                "citations": List[str]       # Specific references
            }
        """
        if not self.llm:
            return None

        try:
            # Build category-specific extraction prompt
            if category == "improvement":
                extraction_focus = """
Extract process improvement insights including best practices, alternative designs, and quantitative metrics.
Include: improvement approach, expected benefits, implementation complexity, relevant KPIs.
"""
            elif category == "comparison":
                extraction_focus = """
Extract alternative architecture/process designs and trade-off analysis.
Include: alternative approach, pros/cons, implementation complexity, comparison factors.
"""
            elif category == "risk_analysis":
                extraction_focus = """
Extract risks, anti-patterns, bottlenecks, and mitigation strategies.
Include: risk description, mitigation approach, compliance considerations, security aspects.
"""
            elif category == "background":
                extraction_focus = """
Extract industry standards, regulations, and domain-specific knowledge.
Include: standard/regulation name, applicability, compliance requirements, industry trends.
"""
            else:
                extraction_focus = "Extract relevant insights for process improvement."

            prompt = f"""
[ROLE]
You are a {domain_source.upper()} solutions architect expert.

[TASK]
{extraction_focus}

[CONTEXT]
- Process: {context.get("process_name", "unknown")}
- Domain: {context.get("domain", "unknown")}
- Source: {domain_source}

[CONTENT]
{raw_content[:3000]}...

[OUTPUT FORMAT]
Return JSON:
{{
    "content": "2-3 sentences in Korean describing the insight",
    "pattern": "Architecture pattern name (e.g., Step Functions Parallel State)",
    "complexity": "Low|Medium|High",
    "citations": ["specific reference 1", "specific reference 2"]
}}

If content is irrelevant, return: {{"content": null}}
"""

            messages = [
                {"role": "system", "content": f"You are a {domain_source} solutions architect."},
                {"role": "user", "content": prompt}
            ]

            response = self.llm.complete(messages)
            text = _extract_text(response).strip()

            # Parse JSON
            result = json.loads(text)

            # Check if result is a dict
            if not isinstance(result, dict):
                LOGGER.warning("[EXTRACT_INSIGHTS][ERROR] Result is not a dict, it's a %s: %s", type(result).__name__, result)
                return None

            if not result.get("content"):
                return None

            return result

        except json.JSONDecodeError as je:
            LOGGER.warning("[EXTRACT_INSIGHTS][ERROR] JSON parse error: %s, text: %s", je, text[:200])
            return None
        except Exception as e:
            LOGGER.exception("[EXTRACT_INSIGHTS][ERROR] %s", e)
            return None

    # ============================================================
    # Utilities
    # ============================================================

    def _make_cache_key(self, query: str, category: str, context: Dict[str, Any]) -> str:
        """
        Generate cache key from query + category + context

        Args:
            query: Search query
            category: Intent category (improvement, comparison, risk_analysis, background)
            context: Process context
        """
        key_str = f"{query}|{category}|{context.get('process_name', '')}|{context.get('domain', '')}"
        return hashlib.md5(key_str.encode()).hexdigest()
