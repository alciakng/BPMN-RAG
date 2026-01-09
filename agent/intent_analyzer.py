# intent_analyzer.py
"""
ReAct Framework Step 1: Query Intent Analysis (LLM-based)
사용자 질의 의도를 LLM을 활용하여 분석, Insight Generation 필요 여부 판단
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from common.logger import Logger
from common.llm_client import LLMClient
from manager.util import _extract_text

LOGGER = Logger.get_logger("agent.intent_analyzer")


class IntentAnalyzer:
    """
    LLM-based query intent analyzer for 2-Stage Agent

    Responsibilities:
    - Determine if external knowledge-based insight generation is needed
    - Provide reasoning for the decision
    """

    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: LLM client for intent classification (required)
        """
        if not llm_client:
            raise ValueError("LLM client is required for IntentAnalyzer")

        self.llm = llm_client

    def analyze(
        self,
        user_query: str,
        stage1_answer: str,
        context_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze query intent using LLM

        Args:
            user_query: Original user query
            stage1_answer: Stage-1 answer from GraphDB
            context_summary: Context summary (optional)

        Returns:
            {
                "needs_insight": bool,           # Insight generation needed?
                "reasoning": str,                # Decision rationale (Korean)
                "confidence": float              # 0.0 ~ 1.0
            }
        """
        try:
            LOGGER.info("[INTENT] Starting LLM-based intent analysis, query_len=%d", len(user_query))

            # LLM-based classification
            result = self._llm_classify_intent(user_query, stage1_answer, context_summary)

            if not result:
                # Fallback if LLM fails
                LOGGER.warning("[INTENT] LLM classification failed, using conservative fallback")
                result = self._fallback_result()

            LOGGER.info(
                "[INTENT] Analysis complete - needs_insight=%s, confidence=%.2f",
                result["needs_insight"],
                result["confidence"]
            )

            return result

        except Exception as e:
            LOGGER.exception("[INTENT][ERROR] %s", e)
            # Conservative fallback: always try insight generation
            return self._fallback_result()

    # ============================================================
    # LLM-based Classification
    # ============================================================

    def _llm_classify_intent(
        self,
        user_query: str,
        stage1_answer: str,
        context_summary: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        LLM-based intent classification
        """
        try:
            system_prompt = """
[ROLE]
You are an expert BPMN Process Intelligence analyst specializing in query intent analysis.

[TASK]
Analyze the user query and Stage-1 answer (GraphDB-based) to determine whether external knowledge-based insight generation is needed.

[DECISION RULES]
**needs_insight = True** if:
- Query asks for recommendations, alternatives, improvements, best practices, or standards
- Query mentions: optimization, efficiency, modernization, cost reduction, industry standards, compliance, automation
- User wants external knowledge (Korean keywords: "개선", "최적화", "업계 사례", "모범 사례", "클라우드", "현대적", "최신", "대안", "비교", "리스크", "병목")
- Query asks: "How to improve?", "What are best practices?", "What are alternatives?", "What are the risks?"
- Query seeks actionable insights beyond simple data retrieval

**needs_insight = False** ONLY if:
- Query is simple information lookup (process structure, flow, participants, data)
- Korean examples: "참여자는 누구인가요?", "프로세스 구조를 설명해주세요", "어떤 데이터를 사용하나요?", "활동은 무엇인가요?"
- Query asks for descriptive information that can be fully answered from GraphDB
- NO mention of improvement, optimization, alternatives, recommendations, or standards

**DEFAULT**: When in doubt, set needs_insight = True
- External knowledge enriches most answers
- Better to have Stage-2 insights than miss them

[OUTPUT FORMAT]
Return ONLY valid JSON (no markdown, no code blocks):
{
    "needs_insight": true,
    "reasoning": "사용자가 프로세스 개선 방법을 요청했으며, 외부 모범 사례와 대안 설계가 도움이 될 것으로 판단됨",
    "confidence": 0.92
}

**IMPORTANT**:
- reasoning must be in Korean (한국어로 작성)
- confidence: 0.0 ~ 1.0 (higher if intent is clear)
"""

            user_message = f"""
[USER QUERY]
{user_query}

[STAGE-1 ANSWER (GraphDB-based)]
{stage1_answer[:1000] if stage1_answer else "No answer yet"}...

[CONTEXT SUMMARY]
{context_summary or "No additional context"}

Analyze the query intent and return JSON.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            # Call LLM
            LOGGER.info("[INTENT][LLM] Calling LLM for intent classification")
            raw_response = self.llm.complete(messages)
            response_text = _extract_text(raw_response)

            LOGGER.info("[INTENT][LLM] Raw response: %s", response_text[:200])

            # Parse JSON
            result = self._parse_intent_response(response_text)

            return result

        except Exception as e:
            LOGGER.exception("[INTENT][LLM][ERROR] %s", e)
            return None

    def _parse_intent_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM JSON response with defensive error handling
        """
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
            if text.startswith("```"):
                # Extract JSON from code block
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

            # Try direct JSON parse
            result = json.loads(text)

            # Validate required fields
            if "needs_insight" not in result:
                raise ValueError("Missing required field: needs_insight")

            # Set defaults
            result.setdefault("reasoning", "")
            result.setdefault("confidence", 0.7)

            # Validate types
            if not isinstance(result["needs_insight"], bool):
                result["needs_insight"] = str(result["needs_insight"]).lower() in ["true", "yes", "1"]

            if not isinstance(result["confidence"], (int, float)):
                result["confidence"] = 0.7

            LOGGER.info("[INTENT][PARSE] Successfully parsed: needs_insight=%s",
                       result["needs_insight"])
            return result

        except json.JSONDecodeError as e:
            LOGGER.warning("[INTENT][PARSE] JSON parse failed: %s, response: %s", e, response_text[:200])

            # Heuristic fallback: keyword matching in response
            lower_text = response_text.lower()
            needs_insight = any(kw in lower_text for kw in [
                '"needs_insight": true',
                '"needs_insight":true',
                'needs_insight=true',
                '개선', '제안', 'improvement'
            ])

            LOGGER.info("[INTENT][PARSE] Heuristic fallback: needs_insight=%s", needs_insight)

            return {
                "needs_insight": needs_insight,
                "reasoning": "LLM 응답 파싱 실패로 인한 휴리스틱 판단",
                "confidence": 0.5
            }

        except Exception as e:
            LOGGER.exception("[INTENT][PARSE][ERROR] Unexpected error: %s", e)
            return None

    def _fallback_result(self) -> Dict[str, Any]:
        """
        Conservative fallback when LLM fails
        """
        return {
            "needs_insight": True,
            "reasoning": "LLM 분석 실패로 인한 보수적 판단 (외부 지식 검색 시도)",
            "confidence": 0.5
        }
