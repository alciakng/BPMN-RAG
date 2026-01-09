# graph_query_agent.py
from __future__ import annotations

import json
import os
import time
import streamlit as st

from agent.context_composer import ContextComposer
from agent.query_interpreter import QueryInterpreter
from agent.intent_analyzer import IntentAnalyzer
from agent.knowledge_augmenter import KnowledgeAugmenter
from agent.answer_aggregator import AnswerAggregator
from manager.reader import Reader
from manager.util import _extract_text
from manager.external_knowledge_store import ExternalKnowledgeStore
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
        knowledge_store: Optional[ExternalKnowledgeStore] = None,
        enable_2stage: bool = True,
    ):
        """
        Args:
            llm_client: Chat/Completion client used to query the LLM.
            reader: Neo4j access layer (Reader) with fully implemented methods.
            context_composer: ContextComposer instance (build_llm_payload).
            interpreter: QueryInterpreter instance (interpret).
            knowledge_store: External knowledge source for Stage-2 (optional).
            enable_2stage: Enable 2-Stage Agent workflow (default: True).
        """
        self.llm = llm_client
        self.reader = reader
        self.composer = context_composer
        self.interpreter = interpreter

        # 2-Stage Agent components
        self.enable_2stage = enable_2stage
        self.knowledge_store = knowledge_store

        if self.enable_2stage:
            self.intent_analyzer = IntentAnalyzer(llm_client=llm_client)
            self.answer_aggregator = AnswerAggregator(llm_client=llm_client)

            if knowledge_store:
                self.knowledge_augmenter = KnowledgeAugmenter(knowledge_store=knowledge_store)
                LOGGER.info("[AGENT] 2-Stage Agent enabled with external knowledge")
            else:
                self.knowledge_augmenter = None
                LOGGER.warning("[AGENT] 2-Stage Agent enabled but no knowledge store provided")
        else:
            LOGGER.info("[AGENT] 2-Stage Agent disabled, using Stage-1 only")

        uri = st.secrets["NEO4J_URI"]
        user = st.secrets["NEO4J_USERNAME"]
        pwd = st.secrets["NEO4J_PASSWORD"]
        db  = st.secrets["NEO4J_DATABASE"]
        api_key  = st.secrets["OPENAI_API_KEY"]

        self.openai_settings = OpenAISettings(api_key=api_key, embedding_model='text-embedding-3-small')
        self.neo4j_settings = Neo4jSettings(uri=uri,username=user,password=pwd,database=db)

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def ingest_and_index_bpmn(
        self,
        file_path: str,
        filename: str,
        container_settings: ContainerSettings,
        parent_category_key: Optional[str] = None,
        predecessor_model_key: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Ingest the BPMN file into Neo4j using bpmn2neo and return {"model_key", "model_name"}.

        Args:
            file_path: Path to BPMN file
            filename: Filename to use as model key
            container_settings: Container settings
            parent_category_key: Parent category key (for CONTAINS_MODEL relationship)
            predecessor_model_key: Predecessor model key (for NEXT_PROCESS relationship)

        Returns:
            Dictionary with model_key and model_name
        """
        try:
            s = Settings(container=container_settings, neo4j=self.neo4j_settings, openai=self.openai_settings)

            modelkey = load_and_embed(
                bpmn_path=file_path,
                model_key=filename,
                settings=s,
                mode='all',
                parent_category_key=parent_category_key,
                predecessor_model_key=predecessor_model_key
            )

            LOGGER.info(
                "[AGENT] bpmn load_and_embed filename=%s, modelKey=%s, parent=%s, predecessor=%s",
                filename, modelkey['model_key'], parent_category_key, predecessor_model_key
            )

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
            modelkey = load_bpmn_to_neo4j(bpmn_path=file_path, model_key=filename, settings=s)
            LOGGER.info("[AGENT] bpmn load filename= %s, modelKey=%s", filename, modelkey)

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
            chat_history: Optional[List[Dict[str, Any]]] = None,
        ) -> str:
            """
            Generate final answer using 2-Stage Agent workflow:

            Stage-1: GraphDB-based answer (existing)
            Stage-2: External knowledge augmentation (new)

            If 2-Stage is disabled or fails, falls back to Stage-1 only.

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

                # 2) Generate Stage-1 answer (GraphDB-based)
                stage1_answer = self._generate_stage1_answer(
                    user_query=user_query,
                    selected_models=selected_models,
                    uploaded_model_key=uploaded_model_key,
                    payload_blocks=payload_blocks,
                    chat_history=chat_history
                )

                if not stage1_answer:
                    LOGGER.warning("[ANSWER] Stage-1 empty; fallback")
                    return "No answer was generated. Please retry with fewer models or a more specific query."

                LOGGER.info("[ANSWER] Stage-1 complete, len=%d", len(stage1_answer))

                # 3) Check if 2-Stage is enabled
                if not self.enable_2stage or not self.knowledge_augmenter:
                    LOGGER.info("[ANSWER] 2-Stage disabled or no knowledge store, returning Stage-1 only")
                    return stage1_answer

                # 4) ReAct Step 1: Intent Analysis
                LOGGER.info("[REACT][STEP1] Starting intent analysis")
                intent_result = self.intent_analyzer.analyze(
                    user_query=user_query,
                    stage1_answer=stage1_answer,
                    context_summary=self._extract_context_summary(payload_blocks)
                )

                LOGGER.info("[REACT][STEP1] needs_insight=%s",
                           intent_result["needs_insight"])

                # 5) ReAct Step 2: External Knowledge Augmentation
                if intent_result["needs_insight"]:
                    LOGGER.info("[REACT][STEP2] Starting knowledge augmentation")

                    process_context = self._build_process_context(
                        payload_blocks=payload_blocks,
                        selected_models=selected_models
                    )

                    augmentation_result = self.knowledge_augmenter.augment(
                        user_query=user_query,
                        stage1_answer=stage1_answer,
                        intent_result=intent_result,
                        process_context=process_context
                    )

                    LOGGER.info("[REACT][STEP2] has_insights=%s, total_sources=%d",
                               augmentation_result["has_insights"],
                               augmentation_result["total_sources"])
                else:
                    LOGGER.info("[REACT][STEP2] Insight not needed, skipping augmentation")
                    augmentation_result = {"has_insights": False, "insights_by_aspect": {}, "total_sources": 0}

                # 6) ReAct Step 3: Answer Aggregation
                LOGGER.info("[REACT][STEP3] Starting answer aggregation")
                final_result = self.answer_aggregator.aggregate(
                    user_query=user_query,
                    stage1_answer=stage1_answer,
                    augmentation_result=augmentation_result,
                    intent_result=intent_result
                )

                LOGGER.info("[REACT][STEP3] Complete - has_stage2=%s", final_result["has_stage2"])

                return final_result["final_answer"]

            except Exception as e:
                LOGGER.exception("[ANSWER][ERROR] 2-Stage workflow failed: %s", e)
                # Attempt Stage-1 fallback
                try:
                    return self._generate_stage1_answer(
                        user_query=user_query,
                        selected_models=selected_models,
                        uploaded_model_key=uploaded_model_key,
                        payload_blocks=payload_blocks,
                        chat_history=chat_history
                    )
                except:
                    return "An error occurred while generating the answer. Please try again."

    def _generate_stage1_answer(
        self,
        user_query: str,
        selected_models: List[str],
        uploaded_model_key: Optional[str],
        payload_blocks: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """
        Generate Stage-1 answer (GraphDB-based, existing logic)
        """
        try:
            # Stage-1 Prompt: Complete analysis including basic improvement recommendations
            SYS_PROMPT_4O = """
                            [GOALS]
                            - If `upload_model_context` exists: primary goal = compare each model in `model_context` vs `upload_model_context` and propose concrete improvements for the uploaded model.
                            - If `upload_model_context` is absent: primary goal = explain the user query with per-model sections using only `model_context` and propose concrete improvements for the model context.

                            [ROLE]
                            - Act as a BPMN/Neo4j Graph-RAG expert and senior Process Innovation Consultant.
                            - Precisely infer the query intent and deliver an accurate answer strictly grounded in the payload.

                            [OPTIONAL — USE ONLY IF QUERY-RELEVANT]

                            - Process Flow Diagram (use when query asks for overall process explanation):
                            Use `model_flows` from the payload to identify predecessor/successor business models, then draw a simple flow diagram with arrows (→).
                            Highlight the current model with **bold** and briefly explain its role within the flow (1-2 sentences).
                            Structure: `[Category]`: predecessor models → **current model** → successor models
                            Example: "`Order Management`: `Order Receipt` → `Credit Check` → **`Inventory Check`** → `Payment Processing` → `Shipping`
                            This model validates stock availability after credit approval and before payment, ensuring order fulfillment readiness."

                            - Problem Diagnosis Table (suggest using a table):
                            Propose a compact table summarizing key issues and evidence.
                            Format: | Issue/Risk | Evidence (IDs, node names) | Impact Area (e.g., lead time/quality/compliance) | Severity (H/M/L) | one-line Solution |

                            - Improvements & Effects Table:
                            For each item include: Action, KPI, baseline → target, expected delta (%), one-line mechanism, risks/assumptions.
                            Example: "lead time ↓15–25% by removing one handoff; first-time-right ↑10–15% by naming gateways + adding receipt ACK".
                            If KPIs or data are insufficient, state it briefly and omit quantification.

                            [RULES]
                            - Use only the payload; if something is missing, say so and propose the minimal patch to collect it.
                            - `CHAT_HISTORY` is reference-only. Do NOT reuse any previous answer verbatim.

                            [STYLE]
                            - Korean only.
                            - Prefer numbered sections, bullet points, and tables. Use bold for extra emphasis.
                            - Do not use HTML (no tags or inline CSS). Prefer pure Markdown for all formatting.
                            - MUST wrap domain terms in inline code (backticks): process/lane/task/data-object/role-title/system names.
                            Examples: `Bank Branch (Front Office)`, `Underwriter`, `credit scoring (bank)`, `request credit score`.
                            limit to 1–3 inline highlights per sentence.

                            [PAYLOAD SCHEMA] (shared by upload_model_context & model_context)
                            {model:{id,name,modelKey,model_flows,parent_category}, participants:[{id,name,properties,
                            processes:[{id,name,modelKey, lanes:[{id,name,properties,flownodes:[{id,name}]}],
                            nodes_all:[{id,name,properties,full_context}], message_flows:[], data_reads:[], data_writes:[],
                            annotations:[], groups:[], lane_handoffs:[], paths_all:[]}
                            ]}]}
                            """

            # keep chat history short for prompt budget
            short_history = (chat_history or [])[-3:]

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

            # Call LLM
            LOGGER.info("[STAGE1][LLM] invoke chat with history")
            text = self._invoke_llm(messages)

            LOGGER.info("[STAGE1][LLM] ok len=%d", len(text) if text else 0)
            return text or ""

        except Exception as e:
            LOGGER.exception("[STAGE1][ERROR] %s", e)
            return ""

    def _extract_context_summary(self, payload_blocks: Dict[str, Any]) -> str:
        """
        Extract brief context summary from payload blocks
        """
        try:
            model_context = payload_blocks.get("model_context", [])
            upload_context = payload_blocks.get("upload_model_context")

            summary_parts = []

            if model_context:
                models = [ctx.get("model", {}).get("name", "Unknown") for ctx in model_context]
                summary_parts.append(f"검색된 모델: {', '.join(models[:3])}")

            if upload_context:
                upload_name = upload_context.get("model", {}).get("name", "Unknown")
                summary_parts.append(f"업로드 모델: {upload_name}")

            return " | ".join(summary_parts) if summary_parts else "No context"

        except Exception as e:
            LOGGER.warning("[CONTEXT_SUMMARY][ERROR] %s", e)
            return "Context extraction failed"

    def _build_process_context(
        self,
        payload_blocks: Dict[str, Any],
        selected_models: List[str]
    ) -> Dict[str, Any]:
        """
        Build process context for knowledge augmentation
        """
        try:
            model_context = payload_blocks.get("model_context", [])

            # Extract process name from first model
            process_name = "business process"
            domain = "general"

            if model_context:
                first_model = model_context[0].get("model", {})
                process_name = first_model.get("name", "business process")

                # Try to extract domain from parent category
                parent_category = first_model.get("parent_category", {})
                if parent_category:
                    domain = parent_category.get("name", "general")

            return {
                "process_name": process_name,
                "domain": domain,
                "selected_models": selected_models,
                "model_summaries": [
                    ctx.get("model", {}).get("name", "Unknown")
                    for ctx in model_context
                ]
            }

        except Exception as e:
            LOGGER.warning("[PROCESS_CONTEXT][ERROR] %s", e)
            return {
                "process_name": "business process",
                "domain": "general",
                "selected_models": [],
                "model_summaries": []
            }

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