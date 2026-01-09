# external_knowledge_store.py
"""
External Knowledge Store Interface
외부 지식 저장소에 대한 통합 인터페이스
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ExternalKnowledgeStore(ABC):
    """
    Abstract interface for external knowledge sources

    Implementations:
    - WebSearchKnowledgeStore: Web search-based (Tavily API)
    - VectorKnowledgeStore: Vector DB-based (future)
    - LLMKnowledgeStore: LLM internal knowledge-based (future)
    """

    @abstractmethod
    def search(
        self,
        query: str,
        aspect: str,
        context: Dict[str, Any],
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Search external knowledge base

        Args:
            query: User query or search intent
            aspect: Knowledge aspect type
                - "alternatives": Alternative process designs
                - "best_practices": Industry best practices
                - "risks": Risk analysis and mitigation
                - "quantitative": Performance metrics and ROI
                - "background": General background knowledge
            context: Process context
                - process_name: str
                - domain: str (e.g., "HR/Payroll")
                - current_issues: List[str]
                - selected_models: List[str]
            top_k: Number of knowledge items to return

        Returns:
            List of knowledge items:
            [
                {
                    "content": str,              # Main content (Korean)
                    "source": str,               # Source title
                    "url": str,                  # Source URL
                    "relevance_score": float,    # 0.0 ~ 1.0
                    "tier": str,                 # "broad" | "focused"
                    "metadata": {
                        "aspect": str,
                        "domain_source": str,    # "apqc" | "aws" | "azure" | "gcp"
                        "architecture_pattern": str,
                        "citations": List[str],
                        ...
                    }
                }
            ]
        """
        pass
