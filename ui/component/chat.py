# ui/component/chat.py
from __future__ import annotations
from pathlib import Path
import tempfile
import streamlit as st
from typing import Dict, Optional, List
from bpmn2neo.settings import ContainerSettings 


from common.logger import Logger
from analytics import track
from ui.app.handler import derive_candidates, answer_with_selected, ingest_and_register_bpmn
from ui.component.panels import render_candidates_selector
from ui.component.chat_input_module import render_chat_input_box
from ui.common.utils import _collect_model_keys
from ui.component.agraph import render_graph_with_selector

LOGGER = Logger.get_logger("ui.chat")

def _ensure_history(welcome_text: str) -> None:
    """
    Seed message history with a single AI welcome message if missing.
    
    Args:
        welcome_text: Welcome message to display on first load
    """
    try:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "ai", "content": welcome_text}]
            LOGGER.info("[CHAT] Message history initialized")
    except Exception as e:
        LOGGER.exception("[CHAT] History initialization failed: %s", str(e))


def _render_input() -> None:
    session_store = st.session_state.get("session_store")
    session_id = st.session_state.get("session_id")

    # Render input box only when no selector is pending
    user_text, uploaded_file, is_submitted = render_chat_input_box(session_id)

    # Handle file upload if present (outside form, so spinner works)
    if uploaded_file is not None:
        existing_uploaded_model = (
            session_store.get_uploaded_model(session_id) 
            if (session_store and session_id) 
            else None
        )

        if existing_uploaded_model is None:
            st.session_state.messages.append({"role": "uploader", "uploaded_file": uploaded_file, "query":user_text})
            st.rerun()
    

    # Process new query only if submitted with valid text
    if not is_submitted:
        LOGGER.debug("[CHAT] No submission, cycle complete")
        return

    # Valid query received
    query = user_text.strip()
    LOGGER.info(
        "[CHAT] Processing query",
        extra={
            "query_length": len(query),
            "session_id": session_id,
            "query_preview": query[:50] + "..." if len(query) > 50 else query
        }
    )

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.last_user_query = query
    st.session_state.trigger_generate = True
    LOGGER.info("[CHAT] Trigger generate : %s", str(st.session_state.trigger_generate))
    
    st.rerun()

def _render_graph() -> None:
    """
    Render the graph section when at least one model key is present.
    - Reads session context from Streamlit's st.session_state
    - Collects model keys via _collect_model_keys(session_store, session_id)
    - Renders a graph panel inside an expander
    """
    try:
        # Log start of rendering step
        LOGGER.info("[GRAPH][RENDER] start")

        # 1) Read session context
        session_store = st.session_state.get("session_store")
        session_id = st.session_state.get("session_id")
        LOGGER.info(
            "[GRAPH][RENDER] session context loaded store=%s, session_id=%s",
            "present" if session_store is not None else "None",
            session_id if session_id is not None else "None",
        )

        # 2) Collect model keys (guard against None)
        try:
            model_keys = _collect_model_keys(session_store, session_id)
            count = len(model_keys) if model_keys else 0
            LOGGER.info("[GRAPH][RENDER] collected model_keys count=%d", count)
        except Exception as e:
            # Explicitly log collection failure without stopping the outer flow
            LOGGER.exception("[GRAPH][RENDER][ERROR] collect model_keys failed: %s", e)
            return

        # 3) Render only when keys exist
        if model_keys and len(model_keys) > 0:
            try:
                with st.expander(label="모델 Graph Flow Diagram"):
                    render_graph_with_selector(model_keys)
                LOGGER.info("[GRAPH][RENDER] graph rendered successfully")
            except Exception as e:
                # Log render failure with traceback
                LOGGER.exception("[GRAPH][RENDER][ERROR] graph render failed: %s", e)
        else:
            LOGGER.info("[GRAPH][RENDER] no model keys; skip rendering")

    except Exception as e:
        # Fatal guard for unexpected errors
        LOGGER.exception("[GRAPH][RENDER][FATAL] %s", e)    

def _has_pending() -> bool:
    """
    Check if there's a pending selector in message history.
    
    Returns:
        bool: True if selector message exists
    """
    try:
        blocking_roles = {"selector", "uploader", "answer", "derive_candidates"}

        for idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            LOGGER.info("[CHAT] role of msg: %s", role)
            if role in blocking_roles:
                return True
        return False
    except Exception as e:
        LOGGER.exception("[CHAT] Error checking pending selector: %s", str(e))
        return False


def _do_task() -> None:
    """
    Render message history including selector UI when present.

    Change:
      - Do NOT save candidates in chat.py. Panels handles persistence.
      - After confirm, just remove selector message and rerun.
    """
    try:
        message_count = len(st.session_state.get("messages", []))
        session_store = st.session_state.get("session_store")
        session_id = st.session_state.get("session_id")

        _render_graph()

        for idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            if role =="derive_candidates":
                with st.spinner("질의대상 프로세스 후보 검색 중..."):
                    # Derive candidates(if selected_model is none)
                    query = st.session_state.get("last_user_query", "")
                    try:
                        LOGGER.info("[CHAT] Deriving candidates")
                        res = derive_candidates(user_query=query)
                        
                        candidates = res.get("candidates") or []
                        
                        LOGGER.info(
                            "[CHAT] Candidates derived len_candidates:%d",len(candidates)
                        )

                        if not candidates:
                            st.session_state.messages.pop(idx)
                            # No candidates found
                            st.session_state.messages.append({
                                "role": "ai",
                                "content": "해당 질의에 대응되는 후보모델이 적재되지 않았습니다. 관리자에게 분석대상 .bpmn 프로세스 적재를 문의하세요."
                            })
                            st.rerun()
                        else:
                            st.session_state.messages.pop(idx)
                            # Add selector message to history
                            st.session_state.messages.append({
                                "role": "selector",
                                "candidates": candidates
                            })
                            st.session_state.trigger_generate = True
                            
                            # Rerun to show selector in chat
                            st.rerun()

                    except Exception as derive_err:
                        LOGGER.exception(
                            "[CHAT] Candidate derivation failed: %s",
                            str(derive_err),
                            extra={"session_id": session_id, "query": query}
                        )
                        st.session_state.messages.append({
                            "role": "ai",
                            "content": "Failed to process query. Please try again."
                        })
                        st.rerun()
            elif role == "selector":
                with st.chat_message("ai"):
                    candidates = msg.get("candidates", [])
                    selected = render_candidates_selector(candidates=candidates)

                    if selected:
                        LOGGER.info(
                            "[CHAT] Selection confirmed from message",
                            extra={"selected_count": len(selected), "models": selected}
                        )
                        
                        # Store selection in session_store
                        if session_store and session_id:
                            # Default to top candidate if none selected
                            if not selected and candidates:
                                selected = [candidates[0]["model_key"]]
                                LOGGER.info(
                                    "[CHAT] No selection, using top candidate",
                                    extra={"model_key": selected[0]}
                                )

                            # Persist selected models for overlay
                            session_store.save_candidates(session_id, selected)

                            LOGGER.info(
                                "[CHAT] Candidates stored in session_store",
                                extra={"session_id": session_id, "models": selected}
                            )
                        
                        st.session_state.messages.pop(idx)
                        st.rerun()
                    return
            elif role == "uploader":
                with st.spinner("업로드 파일 분석중..."):
                    uploaded_file = msg.get("uploaded_file", [])
                    query = msg.get("query", [])
                    upload_info = _process_uploaded_file(uploaded_file, session_id)
                    
                    if upload_info and upload_info.get("model_key"):
                        LOGGER.info(
                            "[CHAT] File upload completed",
                            extra={
                                "model_key": upload_info.get("model_key"),
                                "model_name": upload_info.get("model_name")
                            }
                        )

                        st.session_state.messages.pop(idx)
                        st.success(
                            f"업로드 완료: {upload_info.get('model_name')} "
                            f"(ID: {upload_info.get('model_key')})"
                        )
                        st.session_state.messages.append({"role": "user", "content": query})
                        st.session_state.last_user_query = query
                        st.session_state.trigger_generate = True
                        st.rerun()
            elif role == "answer" :
                with st.spinner("답변 생성 중..."):
                    answer_text = answer_with_selected()

                    if answer_text:
                        st.session_state.messages.pop(idx)

                        # Add answer to message history
                        st.session_state.messages.append({
                            "role": "ai",
                            "content": answer_text,
                        })
                        
                        LOGGER.info(
                            "[CHAT] Answer generated successfully"
                        )
                        st.rerun()
                    else:
                        LOGGER.warning("[CHAT] Empty answer received")
                        st.session_state.messages.append({
                            "role": "ai",
                            "content": "답변이 생성되지 않았습니다. LLM 작업 중 오류가 발생하였습니다."
                        })
            else :
                with st.chat_message(role):
                    st.write(msg.get("content", ""))

        LOGGER.debug("[CHAT] Rendered %d messages", message_count)

    except Exception as e:
        LOGGER.exception("[CHAT][_render_history][ERROR] %s", e)
        st.error("메시지 기록을 렌더링하는 중 오류가 발생했습니다.")


def _process_uploaded_file(uploaded_file, session_id: str) -> Optional[Dict]:
    """
    Process uploaded BPMN file with spinner indication.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        session_id: Current session identifier
        
    Returns:
        dict: Upload info with model_key, or None on failure
        
    Side effects:
        - Creates temporary file
        - Calls ingest_and_register_bpmn
        - Shows spinner during processing
        - Displays success/error messages
        
    Logging:
        - INFO: Processing start and success
        - ERROR: Processing failures
    """
    try:
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        LOGGER.info(
            "[CHAT] Processing uploaded file",
            extra={
                "upload_filename": uploaded_file.name,
                "size_bytes": len(file_bytes),
                "session_id": session_id
            }
        )
        
        # Create temporary file with original extension
        suffix = Path(uploaded_file.name).suffix or ""
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix, 
            prefix="bpmn_"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        # Process with spinner (now works outside form)
        LOGGER.debug(
            "[CHAT] Starting file ingestion",
            extra={
                "filename": uploaded_file.name,
                "temp_path": tmp_path
            }
        )

        # set Container
        container_settings = ContainerSettings(
            create_container=True,
            container_type='upd',
            container_id='updCntr',
            container_name='updBpmn'
        )
        
        upload_info = ingest_and_register_bpmn(
            file_path=tmp_path,
            filename=uploaded_file.name,
            container_settings=container_settings
        )
    
        # Handle result
        if upload_info.get("model_key"):
            
            LOGGER.info(
                "[CHAT] File ingestion successful",
                extra={
                    "model_key": upload_info.get("model_key"),
                    "model_name": upload_info.get("model_name"),
                    "session_id": session_id
                }
            )
            
            return upload_info
        else:
            st.error(" 업로드에 실패했습니다. 파일 형식을 확인해주세요.")
            
            LOGGER.error(
                "[CHAT] File ingestion returned no model_key",
                extra={
                    "upload_filename": uploaded_file.name,
                    "session_id": session_id
                }
            )
            
            return None
            
    except Exception as e:
        LOGGER.exception(
            "[CHAT] File processing error: %s",
            str(e),
            extra={
                "session_id": session_id,
                "upload_filename": getattr(uploaded_file, 'name', 'unknown'),
                "error_type": type(e).__name__
            }
        )
        st.error("❌ 파일 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
        return None

def handle_agent_response() -> None:
    """
    Main UI routine for handling agent interactions.
    Session store based flow control.
    """
    try:

        # Title as plain text (no HTML dependency)
        st.markdown(f"## {'BPMN Graph-RAG'}")

        # Get session context
        session_store = st.session_state.get("session_store")
        session_id = st.session_state.get("session_id")
        agent = st.session_state.get("agent")
        
        if not agent:
            LOGGER.error("[CHAT] Agent not found in session_state")
            st.error("Agent not initialized. Please refresh the page.")
            return
        
        # Initialize session components
        _ensure_history(
            "BPMN 프로세스 기반 프로세스 마이닝 AI-Agent 입니다. "
            "Guide 메뉴를 참고하여 질의를 수행하여 주십시오."
        )
        # Check if selector is pending - if so, hide input box
        has_pending = _has_pending()     

        # Render history (including selector if present)
        _do_task()

        if has_pending:
            LOGGER.info("[CHAT] Selector pending, hiding input box")
            trigger_generate = st.session_state.trigger_generate
            LOGGER.info("[CHAT] Trigger generate : %s", str(trigger_generate))
            # Don't render input box while selector is active
            return
        
        # Verify the `trigger_generate` state set at query submission.
        # Check whether `trigger_generate` (armed on user query) is active.
        if st.session_state.pop("trigger_generate", False):
            # Check if we have selected models in session_store
            selected_objs: List[str] = session_store.get_candidates(session_id) if (session_store and session_id) else []
            LOGGER.info("[CHAT] selected_objs : %d",len(selected_objs))

            # If we have selected models, generate answer
            if selected_objs:
                try:
                    LOGGER.info(
                        "[CHAT] Generating answers"
                    )
                    st.session_state.messages.append({
                            "role": "answer"
                        })
                    st.rerun()
                except Exception as answer_err:
                    LOGGER.exception(
                        "[CHAT] Answer generation failed: %s",
                        str(answer_err)
                    )
                    st.session_state.messages.append({
                        "role": "ai",
                        "content": "Failed to generate answer. Please try again."
                    })
                # Rerun to show the answer
                st.rerun()
        
            # If we don't have selected models, Derive candidates
            else : 
                st.session_state.messages.append({
                    "role": "derive_candidates"
                })
                st.rerun()

        else :
            _render_input()
    
    except Exception as e:
        LOGGER.exception(
            "[CHAT] Critical error in handle_agent_response: %s",
            str(e),
            extra={"session_id": st.session_state.get("SESSION_ID")}
        )
        st.error("An unexpected error occurred. Please refresh the page.")