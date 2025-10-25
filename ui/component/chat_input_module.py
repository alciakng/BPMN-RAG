"""
Pretty ChatGPT-style chat input module for Streamlit.

- Top: multi-line text area (white background)
- Bottom: left = green upload button (BPMN), right = navy send button
"""

from __future__ import annotations

import streamlit as st
from streamlit_extras.bottom_container import bottom
from common.logger import Logger
from streamlit_extras.stateful_button import button

LOGGER = Logger.get_logger("app.handler")

def render_chat_input_box(
    session_id : str,
) -> dict:
    """
    Render a ChatGPT-style input box.

    Args:
        session_id: Unique key prefix for Streamlit widgets.
    Returns:
        dict with:
            - submitted (bool): Whether the Send button was pressed.
            - text (str): The entered text (may be empty).
            - files (List[UploadedFile]): Uploaded files (list, possibly empty).
    """
    
    uploaded_file = None
    text = ""

    try:
        session_store = st.session_state.get("session_store")
        
        # Generate versioned uploader key for reset capability
        uploader_key = (
            session_store.get_uploader_key(session_id) 
            if session_store 
            else "bpmn_uploader_0"
        )

        LOGGER.debug(
            "[INPUT_MODULE] Rendering with dynamic uploader key",
            extra={
                "session_id": session_id,
                "uploader_key": uploader_key
            }
        )

        session_store = st.session_state.get("session_store")
        session_id = st.session_state.get("session_id")
        
        # Get models from session store
        uploaded_model = session_store.get_uploaded_model(session_id) if (session_store and session_id) else None
        
        # Convert uploaded model to list for consistency
        uploaded_models = [uploaded_model] if uploaded_model else None

        # ---------- Layout ----------
        with bottom():
            # Use a form so Enter doesn't submit unexpectedly; explicit Send click
            with st.form(key=f"bpmn-form", clear_on_submit=False):
                text = st.text_area(
                    label="bpmn-rag-input-form",
                    key="bpmn-text",
                    placeholder="프로세스에 대한 질의를 시작해주세요. .bpmn 파일을 업로드하여 질의도 가능합니다.",
                    height=90,
                    label_visibility="collapsed",
                )
                st.markdown('<div class="footer">', unsafe_allow_html=True)
                col_left, col_right = st.columns(
                    [4,1], 
                    gap="small", 
                    vertical_alignment="center",
                    width="stretch"
                )

                with col_left:
                    if uploaded_models is None : 
                        with st.popover(".bpmn 파일업로드", use_container_width=True):
                            # File uploader with dynamic key
                            # Note: Processing happens outside form to enable spinner
                            uploaded_file = st.file_uploader(
                                "Upload BPMN Diagram",
                                type=["bpmn"],
                                accept_multiple_files=False,
                                key=uploader_key,
                                help="Upload a .bpmn file to analyze"
                            )
                            
                            # Display upload status if file exists
                            if uploaded_file is not None:
                                session_store = st.session_state.get("session_store")
                                existing_uploaded_model = (
                                    session_store.get_uploaded_model(session_id) 
                                    if (session_store and session_id) 
                                    else None
                                )
                                
                                if existing_uploaded_model:
                                    st.info("파일이 이미 업로드되어 있습니다.")
                                else:
                                    st.info(f"파일 선택됨: {uploaded_file.name}")
                                    LOGGER.info(
                                        "[INPUT_MODULE] File selected",
                                        extra={
                                            "upload_filename": uploaded_file.name,
                                            "size_bytes": uploaded_file.size,
                                            "session_id": session_id
                                        }
                                    )
                    else :
                        st.info(f'현재 분석중인 업로드 모델 : {uploaded_model}')

                with col_right:
                    if st.form_submit_button(
                        "질의하기", 
                        type="primary", 
                        use_container_width=True
                        ) :

                        return text, uploaded_file, True

                st.markdown("</div>", unsafe_allow_html=True)  # end footer
            st.markdown("</div>", unsafe_allow_html=True)  # end chatbox

        LOGGER.info(
            "[INPUT_MODULE] Render completed",
            extra={
                "text_length": len(text or ""),
                "has_uploaded_file": uploaded_file is not None,
                "session_id": session_id,
                "uploader_key": uploader_key
            }
        )

        return "", None, False

    except Exception as e:
        # Defensive: never crash the page
        LOGGER.exception("render_chat_input_box failed: %s", e)
        st.error("Failed to render the chat input. Please refresh and try again.")
        return  "", None, False


__all__ = ["render_chat_input_box"]
