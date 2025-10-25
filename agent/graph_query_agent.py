# graph_query_agent.py
from __future__ import annotations

import json
import os
import time
import streamlit as st

from agent.context_composer import ContextComposer
from agent.query_interpreter import QueryInterpreter
from manager.reader import Reader
from manager.util import json_dumps_safe, _extract_text
from common.llm_client import LLMClient
from common.logger import Logger
from bpmn2neo.settings import ContainerSettings, OpenAISettings, Neo4jSettings

from typing import Any, Dict, List, Optional


from bpmn2neo import load_and_embed, load_bpmn_to_neo4j # your library
from bpmn2neo.settings import Settings



LOGGER = Logger.get_logger("agent.graph_query_agent")

class GraphQueryAgent:
    """
    Orchestrates the BPMN Graph-RAG workflow end-to-end.

    Responsibilities:
    - Ingest uploaded BPMN and return model identifiers.
    - Derive candidate models (FlowNode-level retrieval -> model-level top-K aggregation).
    - Build per-model LLM payloads and request final answers from the LLM.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        reader: Reader,
        context_composer: ContextComposer,
        interpreter: QueryInterpreter,
    ):
        """
        Args:
            llm_client: Chat/Completion client used to query the LLM.
            reader: Neo4j access layer (Reader) with fully implemented methods.
            context_composer: ContextComposer instance (build_llm_payload).
            interpreter: QueryInterpreter instance (interpret).
            logger: Optional logger (default: module logger).
        """
        self.llm = llm_client
        self.reader = reader
        self.composer = context_composer
        self.interpreter = interpreter

        uri = st.secrets["NEO4J_URI"]
        user = st.secrets["NEO4J_USERNAME"]
        pwd = st.secrets["NEO4J_PASSWORD"]
        db  = st.secrets["NEO4J_DATABASE"]
        api_key  = st.secrets["OPENAI_API_KEY"]

        self.openai_settings = OpenAISettings(api_key=api_key)
        self.neo4j_settings = Neo4jSettings(uri=uri,username=user,password=pwd,database=db)
        

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def ingest_and_index_bpmn(self, file_path: str, filename: str, container_settings: ContainerSettings) -> Dict[str, str]:
        """
        Ingest the BPMN file into Neo4j using bpmn2neo and return {"model_key", "model_name"}.
        """
        try:
            
            s = Settings(container=container_settings, neo4j= self.neo4j_settings,openai=self.openai_settings)
            modelkey = load_and_embed(bpmn_path=file_path, model_key=filename, settings=s,mode='light')
            LOGGER.info("[AGENT] bpmn loaded filename= %s, modelKey=%s", filename, modelkey['model_key'])

            return {"model_key": modelkey['model_key'], "model_name": filename}
        except Exception as e:
            LOGGER.error("[AGENT][ERROR] ingest_and_index_bpmn %s", str(e))
            raise

    def ingest_bpmn(self, file_path: str, filename: str, container_settings: ContainerSettings) -> Dict[str, str]:
        """
        Ingest the BPMN file into Neo4j using bpmn2neo and return {"model_key", "model_name"}.
        """
        try:
            s = Settings(container=container_settings, neo4j= self.neo4j_settings,openai=self.openai_settings)
            modelkey = load_bpmn_to_neo4j(bpmn_path=file_path, model_key=filename,settings=s)
            LOGGER.info("[AGENT] bpmn loaded filename= %s, modelKey=%s", filename, modelkey)

            return {"model_key": modelkey, "model_name": filename}
        except Exception as e:
            LOGGER.error("[AGENT][ERROR] ingest_and_index_bpmn %s", str(e))
            raise

    def derive_candidates_from_query(self, user_query: str, uploaded_model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Derive candidate models and an intent-aware prompt message.

        Returns:
            {
              "candidates": [
                { "model_key": str, "model_name": str,
                  "top_nodes": [ {node_id,..., score, process_id, lane_id} x K ],
                  "score_total": float
                }, ...
              ],
              "prompt_message": str
            }
        """
        try:
            LOGGER.info(
                "[02.CAND] start len(user_query)=%d uploaded=%s",
                len(user_query or ""), bool(uploaded_model_key)
            )
            result = self.interpreter.interpret(user_query=user_query, uploaded_model_key=uploaded_model_key)
            LOGGER.info("[02.CAND] done candidates=%d", len(result.get("candidates", [])))
            return result
        except Exception as e:
            LOGGER.exception("[02.CAND][ERROR] %s", e)
            # Conservative fallback
            return {"candidates": [], "prompt_message": self._fallback_prompt(uploaded_model_key)}

    def answer_with_selected(
            self,
            user_query: str,
            uploaded_model_key: Optional[str] = None,
            chat_history: Optional[List[Dict[str, Any]]] = None,  # NEW: prior Q/A context
        ) -> str:
            """
            Generate final answer using structured payload + prior chat history.

            chat_history: list of {"role": "user"|"assistant", "content": str, "ts": int?}
            is included to provide continuity across turns within the same analysis.
            """
            try:
                session_store = st.session_state.get("session_store")
                session_id = st.session_state.get("session_id")
                selected_models = session_store.get_candidates(session_id) if (session_store and session_id) else []

                # 1) Build payload blocks (now returns {"model_context":[...], "upload_model_context":{...}|None})
                payload_blocks = self.composer.build_llm_payload(
                    uploaded_model_key=uploaded_model_key,
                )
                LOGGER.info("[03.CTX] payload blocks built")

                # 2) Compose messages for LLM
                SYS_PROMPT_4O = """You are a BPMN/Neo4j Graph-RAG Expert and a senior Process Innovation Consultant.
                                GOALS
                                - If `upload_model_context` exists: primary goal = compare each model in `model_context` vs `upload_model_context` and propose concrete improvements for the uploaded model.
                                - If `upload_model_context` is absent: primary goal = explain the user query with per-model sections using only `model_context`.
                                ROLE & INTENT
                                - Act as a senior process innovation consultant with deep domain expertise in the model currently being analyzed. Your goal is to diagnose comprehensive risk, and prescribe practical, high-leverage improvements.
                                KPI Orientation
                                - Tie recommendations to measurable effects (e.g., “reduce lead time 15–25% by removing one handoff; improve first-time-right by naming gateways and adding receipt ACK”).
                                - separate the improvement effect into separate sections with table.
                                STYLE
                                - Prefer tables and numbered sections & bullet points.
                                - Korean only. Do not invent facts outside payload. If info is missing, say so.
                                """

                # keep chat history short for prompt budget
                short_history = (chat_history or [])[-7:]

                def _build_messages(user_query,  selected_models, uploaded_model_key, payload_blocks, short_history):
                    return [
                        {"role": "system", "content": SYS_PROMPT_4O},
                        {"role": "user", "content": f"[QUERY]\n{user_query}\n\n"
                                                    f"[SELECTED_MODELS]\n{selected_models}\n\n"
                                                    f"[UPLOADED_MODEL_KEY]\n{uploaded_model_key or '—'}"},
                        {"role": "user", "content": "[PAYLOAD_BLOCKS]\n" + json.dumps(payload_blocks, ensure_ascii=False)},
                        {"role": "user", "content": "[CHAT_HISTORY]\n" + json.dumps(short_history or [], ensure_ascii=False)},
                    ]
                
                messages = _build_messages(user_query, selected_models, uploaded_model_key, payload_blocks, short_history)

                # full payload -> file (JSON Lines) in current working directory
                _logfile = os.path.join(os.getcwd(), "user_payload.jsonl")
                with open(_logfile, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(messages,
                            ensure_ascii=False,
                            separators=(",", ":"),  # keep compact
                            default=str,
                        )
                    )
                    f.write("\n")

                # 3) Call LLM via completion adapter (_invoke_llm stitches to a single prompt & calls complete())
                LOGGER.info("[04.LLM] invoke chat with history")
                text = self._invoke_llm(messages)
                if not text:
                    LOGGER.warning("[04.LLM] empty response; fallback")
                    return "No answer was generated. Please retry with fewer models or a more specific query."

                LOGGER.info("[04.LLM] ok len=%d", len(text))
                return text
            except Exception as e:
                LOGGER.exception("[04.LLM][ERROR] %s", e)
                return "An error occurred while generating the answer. Please try again."

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _invoke_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Invoke the LLM client in a defensive but minimal way:
        - Prefer .chat(messages)
        - Fallback to .complete(prompt) if available
        - Extract text across common response shapes
        """
        try:
            # chat-style
            if hasattr(self.llm, "complete"):
                resp = self.llm.complete(messages)
                return _extract_text(resp)

            # no supported interface
            LOGGER.error("[LLM][ERROR] client has no 'chat' or 'complete' method")
            return ""
        except Exception as e:
            LOGGER.exception("[LLM][INVOKE][ERROR] %s", e)
            return ""

    def _fallback_prompt(self, uploaded_model_key: Optional[str]) -> str:
        """Conservative developer prompt fallback."""
        if uploaded_model_key:
            return ("Write a structured comparison answer by model sections; "
                    "include commonalities, differences, risks, and recommendations.")
        return ("Write a structured answer by model sections if multiple models; "
                "explain end-to-end flows, lane handoffs, message flows, and data I/O.")