"""
Pretty ChatGPT-style chat input module for Streamlit.

- Top: multi-line text area (white background)
- Bottom: left = green upload button (BPMN), right = navy send button
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import streamlit as st
from common.logger import Logger

from streamlit_extras.bottom_container import bottom
from streamlit_extras.stylable_container import stylable_container

LOGGER = Logger.get_logger("app.handler")


def render_chat_input_box(
    session_id: str,
) -> dict:
    """
    Render a bottom-fixed chat input using st.chat_input only.

    Behavior:
      - Allows file attachments (".bpmn" only).
      - When the user submits, returns (text, uploaded_file, True).
      - Otherwise returns ("", None, False).
      - Disables input while st.session_state.trigger_generate is True.
      - Shows a textual label next to the upload icon via CSS if native label is unavailable.
    """
    try:
        # Initialize states once
        st.session_state.setdefault("trigger_generate", False)
        disabled = bool(st.session_state.trigger_generate)

        # Generate versioned uploader key for reset capability
        session_store = st.session_state.get("session_store")
        uploader_key = (
            session_store.get_uploader_key(session_id) 
            if session_store 
            else "bpmn_uploader_0"
        )

        # Get models from session store
        uploaded_model = session_store.get_uploaded_model(session_id) if (session_store and session_id) else None
        # Convert uploaded model to list for consistency
        uploaded_models = [uploaded_model] if uploaded_model else None

        # CSS fallback: append a small label next to the upload icon
        # Note: aria-label and test ids can vary by Streamlit version.
        st.markdown(
            """
            <style>
                section[data-testid="stChatInput"] { max-width: 100% !important; }
                section[data-testid="stChatInput"] button[aria-label="Upload a file"]::after {
                    content: " .bpmn 파일업로드";
                    font-size: 12px;
                    margin-left: 6px;
                    opacity: 0.9;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Common kwargs for chat_input
        kwargs = dict(
            key="bpmn-text",
            placeholder="프로세스에 대한 질의를 시작해주세요. .bpmn 파일을 업로드하여 질의도 가능합니다.",
            disabled=disabled,
        )

        # ---- Normalize submission payload ----
        text: str = ""
        uploaded_file = None
        submitted: bool = False

        with bottom():
            if prompt := st.chat_input(
                        **kwargs,
                        accept_file=True,
                        file_type=["bpmn"]
                    ):
                
                if prompt and prompt.text:
                    text = prompt.text

                    if prompt and prompt["files"]:
                        uploaded_file = prompt["files"][0]

                    submitted = True

                # Lock input after submission until the outer processing resets the flag
                if submitted and not disabled:
                    st.session_state.trigger_generate = True

                try:
                    LOGGER.info(
                        "[INPUT_MODULE] Render completed | text_length=%s | has_uploaded_file=%s | session_id=%s | key=%s",
                        text,
                        bool(uploaded_file),
                        session_id,
                        "bpmn-text",
                    )
                except Exception:
                    pass

                # Contract: return (text, uploaded_file, True) if submitted, otherwise defaults
                if submitted:
                    return text, uploaded_file, True
                else:
                    return "", None, False
                
            # Optional informational banner for current uploaded model
            if uploaded_models is not None:
                st.info(f"현재 분석중인 업로드 모델 : {uploaded_model}")
        
    except Exception as e:
        # Defensive: never crash the page
        try:
            LOGGER.exception("render_chat_input_box failed: %s", e)
        except Exception:
            pass
        st.error("채팅 입력을 렌더링하지 못했습니다. 새로고침 후 다시 시도해주세요.")
        return "", None, False

    return "", None, False

__all__ = ["render_chat_input_box"]
