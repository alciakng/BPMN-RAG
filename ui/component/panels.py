# ui/component/panels.py
import json
import streamlit as st
from typing import Dict, List, Optional
from common.logger import Logger
from ui.app.handler import reset_candidates, reset_history
from ui.app.handler import reset_uploaded_model

LOGGER = Logger.get_logger("ui.panels")

@st.dialog("모델 초기화 확인")
def confirm_reset_dialog(reset_type="candidates"):
    """Reset confirmation dialog."""
    message = "분석중인 모델이 초기화됩니다." if reset_type == "candidates" else "업로드된 모델이 초기화됩니다."
    st.info(f"⚠️ {message} 계속하시겠습니까?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("취소", use_container_width=True, key=f"cancel_reset_{reset_type}"):
            st.rerun()
    with col2:
        if st.button("확인", type="primary", use_container_width=True, key=f"confirm_reset_{reset_type}"):
            # clear history
            reset_history()
            if reset_type == "candidates":
                reset_candidates()
            else:
                reset_uploaded_model()
            st.success("✓ 모델이 초기화되었습니다.")
            st.rerun()

def render_selected_models_sidebar() -> None:
    """
    Render selected models list and reset button in sidebar.
    Shows two sections: analysis models and uploaded models.
    """
    try:
        session_store = st.session_state.get("session_store")
        session_id = st.session_state.get("session_id")
        
        # Get models from session store
        selected_models = session_store.get_candidates(session_id) if (session_store and session_id) else []
        uploaded_model = session_store.get_uploaded_model(session_id) if (session_store and session_id) else None
        
        # Convert uploaded model to list for consistency
        uploaded_models = [uploaded_model] if uploaded_model else []
        
        # Only render if there are any models
        if not selected_models and not uploaded_models:
            LOGGER.debug("[PANELS][SIDEBAR] No models, not rendering section")
            return
        
        st.sidebar.markdown("---")
        
        # Section 1: Analysis Models
        if selected_models:
            with st.sidebar.container(border=True):
                st.markdown("### 📊 분석중인 모델")
                
                # Custom HTML list for analysis models
                models_html = "".join([
                    f"""
                    <div style='padding: 10px 15px; margin-bottom: 10px; background: #ffffff; 
                                border-left: 3px solid #1f6feb; border-radius: 6px;'>
                        <span style='color: #000000; font-size: 14px;'>▪ {model}</span>
                    </div>
                    """ 
                    for model in selected_models
                ])
                
                st.markdown(models_html, unsafe_allow_html=True)
                
                LOGGER.info(
                    "[PANELS][SIDEBAR] Displaying analysis models",
                    extra={"count": len(selected_models), "models": selected_models}
                )
                
                # Reset button for analysis models
                if st.button(
                    "분석모델 초기화",
                    key="reset_analysis_btn",
                    help="선택된 분석 모델을 초기화합니다",
                    use_container_width=True,
                    type="primary"
                ):
                    confirm_reset_dialog("candidates")
        
        # Section 2: Uploaded Models
        if uploaded_models:
            with st.sidebar.container(border=True):
                st.markdown("### 📁 업로드된 모델")
                
                # Custom HTML list for uploaded models
                upload_html = "".join([
                    f"""
                    <div style='padding: 8px 12px; margin-bottom: 6px; background: #262730; 
                                border-left: 3px solid #2ecc71; border-radius: 4px;'>
                        <span style='color: #ffffff; font-size: 14px;'>▪ {model}</span>
                    </div>
                    """ 
                    for model in uploaded_models
                ])
                
                st.markdown(upload_html, unsafe_allow_html=True)
                
                LOGGER.info(
                    "[PANELS][SIDEBAR] Displaying uploaded models",
                    extra={"count": len(uploaded_models), "models": uploaded_models}
                )
                
                # Reset button for uploaded models
                if st.button(
                    "업로드모델 초기화",
                    key="reset_upload_btn",
                    help="업로드된 모델을 초기화합니다",
                    use_container_width=True,
                    type="primary"
                ):
                    confirm_reset_dialog("uploaded")
                
    except Exception as e:
        LOGGER.exception("[PANELS][SIDEBAR] Error: %s", str(e))
        st.sidebar.error("모델 목록을 불러오는 중 오류가 발생했습니다.")


def render_candidates_selector(
    candidates: List[Dict],
    title: str = "분석할 모델 선택",
    open_button_label: Optional[str] = "분석모델 선택",
    session_ns: str = "cand",
) -> Optional[List[str]]:
    """
    Controller that opens a dialog for candidate selection using tree_select.

    Args:
        candidates: Category hierarchy tree in streamlit_tree_select format
            [
                {
                    "label": "Category Name",
                    "value": "category_key",
                    "children": [
                        {"label": "Model Name", "value": "model_key"}
                    ]
                }
            ]
        title: Dialog title
        open_button_label: Optional button label to open dialog
        session_ns: Session namespace prefix

    Returns:
        List of selected model keys (leaf nodes only) or None
    """
    try:
        # --- 0) Guard
        if not candidates:
            st.info("선택할 후보 모델이 없습니다.")
            return None

        # Import tree_select
        try:
            from streamlit_tree_select import tree_select
        except ImportError:
            LOGGER.error("[CAND][TREE] streamlit_tree_select not installed")
            st.error("streamlit_tree_select 라이브러리가 설치되지 않았습니다. `pip install streamlit-tree-select`를 실행하세요.")
            return None

        # --- 1) Build unique namespace
        # Extract all leaf model keys for hash
        def extract_model_keys(nodes):
            keys = []
            for node in nodes:
                if not node.get("_is_category", True):  # Leaf node (model)
                    if "value" in node:
                        keys.append(node["value"])
                if "children" in node:
                    keys.extend(extract_model_keys(node["children"]))
            return keys

        all_model_keys = extract_model_keys(candidates)
        if not all_model_keys:
            st.warning("후보 모델에 선택 가능한 모델이 없습니다.")
            return None

        key_suffix = str(abs(hash(tuple(sorted(all_model_keys)))))
        ns = f"{session_ns}_{key_suffix}"

        # --- 2) Session keys
        open_flag_key = f"{ns}_open"
        confirmed_key = f"{ns}_confirmed"

        # Set default open
        st.session_state[open_flag_key] = True

        # --- Optional open button
        if open_button_label:
            if st.button(open_button_label, key=f"{ns}_open_btn", type="primary", use_container_width=True):
                st.session_state[open_flag_key] = True

        # --- 3) Define dialog (modal) once
        @st.dialog(title, width="large")
        def _modal():
            """Dialog UI with tree_select."""
            try:
                st.success("질의에 매칭되는 후보모델을 식별하였습니다. 카테고리별로 분석을 진행할 모델을 선택하세요. (복수선택 가능)")
                st.write("")

                # Tree select component (clean, no custom styling)
                selected = tree_select(
                    candidates,  # First positional argument is the tree data
                    expanded=[node["value"] for node in candidates],  # Expand all categories
                    show_expand_all=True,  # Show expand/collapse all button
                    key=f"{ns}_tree"
                )

                LOGGER.info("[CAND][TREE] Tree selection result: %s", selected)

                # Confirmation button
                st.write("")
                if st.button("✓ 확인", key=f"{ns}_ok", type="primary", use_container_width=True):
                    # Extract selected model keys (leaf nodes only)
                    chosen = selected.get("checked", []) if selected else []

                    # Filter only model keys (exclude category keys)
                    model_keys_only = [k for k in chosen if k in all_model_keys]

                    LOGGER.info(
                        "[CAND][CONFIRM] %d models selected from tree: %s",
                        len(model_keys_only),
                        model_keys_only
                    )

                    # Save result, close dialog on next run
                    st.session_state[confirmed_key] = model_keys_only
                    st.session_state[open_flag_key] = False
                    st.rerun()
                
                

            except Exception as e:
                LOGGER.exception("[CAND][DIALOG] modal failed: %s", e)
                st.error("후보 선택 팝업을 표시하는 중 오류가 발생했습니다.")

        # --- 4) Open dialog if flag set
        if st.session_state.get(open_flag_key, False):
            _modal()

        # --- 5) If confirmed on the previous run, return it now (and clear)
        chosen: Optional[List[str]] = st.session_state.pop(confirmed_key, None)
        if chosen:
            return chosen

        return None

    except Exception as e:
        LOGGER.exception("[CAND][DIALOG] render_candidates_selector failed: %s", e)
        st.error("후보 선택 팝업을 표시하는 중 오류가 발생했습니다.")
        return None






