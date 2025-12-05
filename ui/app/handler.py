# handler.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from manager.reader import Reader
from manager.vector_store import FaissStore
from common.logger import Logger
from common.util import filename_key, upload_image_to_s3
from bpmn2neo.settings import ContainerSettings

from agent.graph_query_agent import GraphQueryAgent
from manager.session_store import SessionStore


LOGGER = Logger.get_logger("app.handler")

def _get_agent() -> GraphQueryAgent:
    return st.session_state.get("agent")

def _get_reader() -> Reader:
    return st.session_state.get("reader")

def _get_session_store() -> SessionStore:
    return st.session_state.get("session_store")

def _get_vector_store() -> FaissStore:
    return st.session_state.get("vector_store")

def _get_session_id() -> Optional[str]:
    return st.session_state.get("session_id")

# ----------------------------- BPMN ingest -----------------------------
def ingest_and_register_bpmn(
    file_path: str,
    filename: str,
    container_settings: ContainerSettings,
    agent: Optional[GraphQueryAgent] = None,
    session_store: Optional[SessionStore] = None,
    session_id: Optional[str] = None,
    parent_category_key: Optional[str] = None,
    predecessor_model_key: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Ingest a BPMN file, index it, and optionally register the uploaded model key.

    Args:
        file_path: Path to BPMN file
        filename: Filename to use as model key
        container_settings: Container settings
        agent: GraphQueryAgent instance (optional)
        session_store: SessionStore instance (optional)
        session_id: Session ID (optional)
        parent_category_key: Parent category key (for CONTAINS_MODEL relationship)
        predecessor_model_key: Predecessor model key (for NEXT_PROCESS relationship)

    Returns:
        Dictionary with model_key and model_name
    """
    try:
        # Resolve dependencies
        ag = agent or _get_agent()
        store = session_store or _get_session_store()
        sid = session_id or _get_session_id()

        if ag is None:
            LOGGER.error("[HANDLER][INGEST] agent unavailable; abort %s",
                         json.dumps({"filename": filename, "file_path": file_path, "session_id": sid}, ensure_ascii=False))
            return {"model_key": None, "model_name": None}

        LOGGER.info("[HANDLER][INGEST] start %s",
                    json.dumps({
                        "filename": filename,
                        "file_path": file_path,
                        "session_id": sid,
                        "parent_category_key": parent_category_key,
                        "predecessor_model_key": predecessor_model_key
                    }, ensure_ascii=False))

        # --- decision based on container_settings.type / container_type ---
        ctype = None
        try:
            if hasattr(container_settings, "type"):
                ctype = getattr(container_settings, "type")
            elif hasattr(container_settings, "container_type"):
                ctype = getattr(container_settings, "container_type")
            elif isinstance(container_settings, dict):
                ctype = container_settings.get("type") or container_settings.get("container_type")
        except Exception:
            ctype = None

        ctype_str = (str(ctype).strip().lower()) if ctype is not None else None
        persist = (ctype_str == "upd")

        # Ingest with category and predecessor
        info = ag.ingest_and_index_bpmn(
            file_path=file_path,
            filename=filename,
            container_settings=container_settings,
            parent_category_key=parent_category_key,
            predecessor_model_key=predecessor_model_key
        ) or {}

        model_key = info.get("model_key")
        model_name = info.get("model_name")

        if not model_key:
            LOGGER.warning("[HANDLER][INGEST] model_key is None %s",
                           json.dumps({"filename": filename}, ensure_ascii=False))
            return {"model_key": model_key, "model_name": model_name}

        if persist:
            if store and sid:
                try:
                    store.save_uploaded_model(sid, model_key)
                    LOGGER.info("[HANDLER][INGEST] uploaded model_key stored %s",
                                json.dumps({"model_key": model_key, "session_id": sid, "ctype": ctype_str}, ensure_ascii=False))
                except Exception as se:
                    LOGGER.warning("[HANDLER][INGEST] persist failed %s",
                                   json.dumps({"err": str(se), "model_key": model_key, "session_id": sid, "ctype": ctype_str}, ensure_ascii=False))
            else:
                LOGGER.info("[HANDLER][INGEST] skip persist (no session store/id) %s",
                            json.dumps({"model_key": model_key, "session_id": sid, "ctype": ctype_str}, ensure_ascii=False))
        else:
            LOGGER.info("[HANDLER][INGEST] do not persist (container type not 'upd') %s",
                        json.dumps({"model_key": model_key, "session_id": sid, "ctype": ctype_str}, ensure_ascii=False))

        return {"model_key": model_key, "model_name": model_name}

    except Exception as e:
        LOGGER.exception("[HANDLER][INGEST][ERROR] %s", str(e))
        return {"model_key": None, "model_name": None}

# ----------------------------- Candidates ------------------------------
def derive_candidates(user_query: str) -> Dict[str, Any]:
    try:
        agent = _get_agent()
        session_store = _get_session_store()
        session_id = _get_session_id()
        uploaded_model_key = session_store.get_uploaded_model(session_id)

        LOGGER.info("[HANDLER][CAND] query len=%d uploaded=%s", len(user_query or ""), bool(uploaded_model_key))
        result = agent.derive_candidates_from_query(user_query=user_query, uploaded_model_key=uploaded_model_key)

        return result
    except Exception as e:
        LOGGER.exception("[HANDLER][CAND][ERROR] %s", e)
        return {"candidates": [], "prompt_message": "An error occurred while deriving candidates."}

# ------------------------------- Answer --------------------------------
def answer_with_selected() -> str:
    """
    Generate final answer using selected Model OBJECTS persisted in SessionStore.

    Changes:
      - No parameters. Read candidates from session_store.get_candidates(session_id).
      - Build analysis lifecycle using model_keys extracted from objects.
      - Use prompt_message from session_state['last_prompt_message'] if available; fallback otherwise.
      - Implements context offloading: aggregates recent history (limit 1) with
        vector-similar Q&A pairs (top 2) from FAISS store.
    """
    try:
        agent = _get_agent()
        session_store = _get_session_store()
        vector_store = _get_vector_store()
        session_id = _get_session_id()

        uploaded_model_key = session_store.get_uploaded_model(session_id)
        user_query = st.session_state.get("last_user_query", "")

        selected_models: List[str] = session_store.get_candidates(session_id) if (session_store and session_id) else []

        # Analysis id lifecycle
        cur_analysis = session_store.get_current_analysis(session_id)
        reuse = False
        if cur_analysis:
            bound_models = set(session_store.get_analysis_models(session_id, cur_analysis))
            if bound_models == set(selected_models):
                reuse = True

        if not reuse:
            analysis_id = session_store.create_analysis(session_id, selected_models)
        else:
            analysis_id = cur_analysis

        # Build chat_history and append user turn
        session_store.append_history(session_id, analysis_id, role="user", content=user_query)

        # Context offloading: Aggregate recent + vector-similar history
        # 1. Get most recent Q&A pair (limit=1 from session_store)
        recent_history = session_store.get_history(session_id, analysis_id, limit=1)

        # 2. Search for similar Q&A pairs from vector store (top 2)
        similar_history = []
        if vector_store:
            try:
                similar_pairs = vector_store.search_similar(
                    session_id=session_id,
                    query=user_query,
                    top_k=2
                )

                # Convert similar pairs to chat history format
                for pair in similar_pairs:
                    similar_history.append({
                        "role": "user",
                        "content": pair["query"],
                        "ts": pair["timestamp"]
                    })
                    similar_history.append({
                        "role": "assistant",
                        "content": pair["answer"],
                        "ts": pair["timestamp"]
                    })

                LOGGER.info("[HANDLER][ANS] Found %d similar Q&A pairs from vector store", len(similar_pairs))
            except Exception as ve:
                LOGGER.warning("[HANDLER][ANS] Vector search failed: %s", ve)

        # 3. Aggregate: similar history + recent history (chronologically)
        aggregated_history = similar_history + recent_history

        # Sort by timestamp if available
        aggregated_history.sort(key=lambda x: x.get("ts", 0))

        LOGGER.info(
            "[HANDLER][ANS] Context offloading: recent=%d similar=%d aggregated=%d",
            len(recent_history), len(similar_history), len(aggregated_history)
        )

        # Ask agent with aggregated history
        text = agent.answer_with_selected(
            user_query=user_query,
            uploaded_model_key=uploaded_model_key,
            chat_history=aggregated_history,
        )

        # Append assistant turn to session store
        session_store.append_history(session_id, analysis_id, role="assistant", content=text)
        LOGGER.info("[HANDLER][ANS] stored assistant turn len=%d", len(text or ""))

        # Save Q&A pair to vector store for future similarity search
        if vector_store:
            try:
                success = vector_store.add_qa_pair(
                    session_id=session_id,
                    query=user_query,
                    answer=text
                )
                if success:
                    LOGGER.info("[HANDLER][ANS] Saved Q&A pair to vector store")
                else:
                    LOGGER.warning("[HANDLER][ANS] Failed to save Q&A pair to vector store")
            except Exception as ve:
                LOGGER.exception("[HANDLER][ANS] Error saving to vector store: %s", ve)

        return text
    except Exception as e:
        LOGGER.exception("[HANDLER][ANS][ERROR] %s", e)
        return "An error occurred while generating the answer."


# ------------------------------ Reset/Clear ----------------------------
def reset_history() -> None:
    """
    Clear only the chat history of the current analysis.
    Behavior:
      - If model/analysis exists: clear history and notify user.
      - If no current analysis: log warning and notify user.
      - On any error: log exception; best-effort user notice via system_notice.
    """
    try:
        session_store = _get_session_store()
        session_id = _get_session_id()

        cur_analysis = session_store.get_current_analysis(session_id)
        if not cur_analysis:
            LOGGER.warning("[HANDLER][RESET_HISTORY] no current analysis for session %s", session_id)
            return

        # Clear chat history for the current analysis
        session_store.clear_history(session_id, cur_analysis)
        LOGGER.info(
            "[HANDLER][RESET_HISTORY] cleared history for session %s analysis %s",
            session_id, cur_analysis
        )

        # Push an AI message into chat history
        try:
            st.session_state.messages.append({
                "role": "ai",
                "content": "대화 기록이 초기화되었습니다. 계속 진행하려면 메시지를 입력하세요."
            })
        except Exception as ee:
            LOGGER.exception("[HANDLER][RESET_HISTORY][HISTORY][ERROR] %s", ee)
            st.session_state.system_notice = "대화 기록이 초기화되었습니다."

    except Exception as e:
        LOGGER.exception("[HANDLER][RESET_HISTORY][ERROR] %s", e)



def reset_candidates() -> None:
    """
    Clear selected models, delete current analysis and its history,
    and push an assistant line to the chat history so it shows in chat window.
    """
    try:
        session_store = _get_session_store()
        session_id = _get_session_id()

        # clear selected models
        session_store.clear_candidates(session_id)

        # delete current analysis (models + history)
        cur = session_store.get_current_analysis(session_id)
        if cur:
            session_store.delete_analysis(session_id, cur)

        LOGGER.info("[HANDLER][RESET] cleared candidates & deleted current analysis for session %s", session_id)

        # Push an AI message into chat history using chat._ensure_history
        try:
             st.session_state.messages.append({"role": "ai", "content": '분석중인 모델이 초기화되었습니다. 새로운 질의를 시작하세요.'})
        except Exception as ee:
            LOGGER.exception("[HANDLER][RESET][HISTORY][ERROR] %s", ee)
            st.session_state.system_notice = "분석중인 모델이 초기화되었습니다. 새로운 질의를 시작하세요."
    except Exception as e:
        LOGGER.exception("[HANDLER][RESET][ERROR] %s", e)


def reset_uploaded_model() -> None:
    """
    Clear uploaded model from session.
    """
    try:
        session_store = _get_session_store()
        session_id = _get_session_id()

        # Clear uploaded model
        session_store.clear_uploaded_model(session_id)  # 또는 적절한 메소드명 사용

        LOGGER.info("[HANDLER][RESET_UPLOAD] cleared uploaded model for session %s", session_id)

        # Push an AI message into chat history
        try:
            st.session_state.messages.append({"role": "ai", "content": '업로드된 모델이 삭제되었습니다. 새로운 파일을 업로드하여 분석을 진행하세요.'})
        except Exception as ee:
            LOGGER.exception("[HANDLER][RESET_UPLOAD][HISTORY][ERROR] %s", ee)
            st.session_state.system_notice = "업로드된 모델이 초기화되었습니다."
    except Exception as e:
        LOGGER.exception("[HANDLER][RESET_UPLOAD][ERROR] %s", e)


def handle_image_upload_to_s3(image_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Upload optional image to S3 using Streamlit secrets.
    Key = filename without extension. Returns (object_key, public_url).
    """
    try:
        obj_key = filename_key(filename)
        url = upload_image_to_s3(image_bytes=image_bytes, object_key=obj_key, filename=filename)
        LOGGER.info("[UPLOAD][IMG] Uploaded key=%s url=%s", obj_key, url)
        return obj_key, url
    except Exception as e:
        LOGGER.exception("[UPLOAD][IMG][ERROR] %s", e)
        raise

    
def fetch_graph_for_tabs(model_key: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Fetch graph datasets for 4 tabs via Reader.
    Returns:
        {
            "overall": {"nodes":[...], "edges":[...]},
            "message": {"nodes":[...], "edges":[...]},
            "subprocess":{"nodes":[...], "edges":[...]},
            "dataio":   {"nodes":[...], "edges":[...]},
        }
    """
    reader = _get_reader()
    key_primary = model_key or "default"

    try:
        LOGGER.info("[UPLOAD][GRAPH] fetch graphs model_key=%s", key_primary)

        # Try with primary key
        overall = reader.get_overall_process_graph(key_primary)

        # Fallback condition #2: overall is missing/empty → retry with 'default'
        if not overall and key_primary != "default":
            LOGGER.warning("[UPLOAD][GRAPH] overall empty for key=%s; retry with 'default'", key_primary)
            key_primary = "default"
            
        overall = reader.get_overall_process_graph(key_primary)
        message = reader.get_message_exchange_graph(key_primary)
        subprocess = reader.get_subprocess_graph(key_primary)
        dataio = reader.get_data_io_graph(key_primary)

        return {
            "overall": overall,
            "message": message,
            "subprocess": subprocess,
            "dataio": dataio,
        }

    except Exception as e:
        # Fallback condition #3: any error → retry with 'default'
        LOGGER.exception("[UPLOAD][GRAPH][ERROR] %s", e)
        try:
            LOGGER.info("[UPLOAD][GRAPH] fallback to 'default'")
            overall = reader.get_overall_process_graph("default")
            message = reader.get_message_exchange_graph("default")
            subprocess = reader.get_subprocess_graph("default")
            dataio = reader.get_data_io_graph("default")
            return {
                "overall": overall,
                "message": message,
                "subprocess": subprocess,
                "dataio": dataio,
            }
        except Exception as e2:
            LOGGER.exception("[UPLOAD][GRAPH][FALLBACK-ERROR] %s", e2)
            raise