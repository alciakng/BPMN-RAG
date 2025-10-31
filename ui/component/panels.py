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
                    <div style='padding: 10px 15px; margin-bottom: 10px; background: #262730; 
                                border-left: 3px solid #1f6feb; border-radius: 6px;'>
                        <span style='color: #ffffff; font-size: 14px;'>▪ {model}</span>
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
    Controller that opens a dialog for candidate selection and returns the chosen keys.
    Contract:
      - Returns List[str] after user confirms in dialog.
      - Returns None if not confirmed yet / canceled / closed.
    """
    try:
        # --- 0) Guard
        if not candidates:
            st.info("선택할 후보 모델이 없습니다.")
            return None

        # --- 1) Build stable keys & labels
        model_keys: List[str] = []
        labels: List[str] = []
        for c in candidates:
            mk = (c.get("model_key") or "").strip()
            mn = (c.get("model_name") or mk) or "Unnamed"
            if mk:
                model_keys.append(mk)
                labels.append(f"{mn} · {mk}")

        if not model_keys:
            st.warning("후보 모델에 model_key가 없습니다.")
            return None

        key_suffix = str(abs(hash(tuple(model_keys))))
        ns = f"{session_ns}_{key_suffix}"

        # --- 2) Session keys
        open_flag_key      = f"{ns}_open"
        selected_state_key = f"{ns}_selected"   # temp set of selected keys
        confirmed_key      = f"{ns}_confirmed"  # final chosen list

        # set Default True
        st.session_state[open_flag_key] = True
        # Ensure state container
        if selected_state_key not in st.session_state:
            st.session_state[selected_state_key] = set()

        # --- Optional open button
        if open_button_label:
            if st.button(open_button_label, key=f"{ns}_open_btn", type="primary", use_container_width=True):
                st.session_state[open_flag_key] = True

        # --- 3) Define dialog (modal) once
        @st.dialog(title, width="small")
        def _modal():
            """Dialog UI – cannot return to caller; must set state and rerun."""
            try:
                st.success("질의에 매칭되는 후보모델을 식별하였습니다. 분석을 진행할 모델을 선택하세요. (복수선택 가능)")
                st.write("")

                list_height = 320
                # Checkbox list
                with st.container(height=list_height, border=True):
                    for mk, label in zip(model_keys, labels):
                        ckey = f"{ns}_chk_{mk}"
                        checked = mk in st.session_state[selected_state_key]
                        new_val = st.checkbox(label, key=ckey, value=checked)
                        if new_val:
                            st.session_state[selected_state_key].add(mk)
                        else:
                            st.session_state[selected_state_key].discard(mk)


                if st.button("✓ 확인", key=f"{ns}_ok", type="primary", use_container_width=True):
                    chosen = sorted(list(st.session_state[selected_state_key]))
                    LOGGER.info(
                        "[CAND][CONFIRM] %d selected", len(chosen),
                        extra={"selected": chosen, "count": len(chosen)}
                    )
                    # Save result, close dialog on next run
                    st.session_state[confirmed_key] = chosen
                    st.session_state[open_flag_key] = False
                    st.rerun()

            except Exception as e:
                LOGGER.exception("[CAND][DIALOG] modal failed: %s", e)
                st.error("후보 선택 팝업을 표시하는 중 오류가 발생했습니다.")

        # --- 5) Open dialog if flag set
        if st.session_state.get(open_flag_key, False):
            _modal()

        # --- 6) If confirmed on the previous run, return it now (and clear)
        chosen: Optional[List[str]] = st.session_state.pop(confirmed_key, None)
        if chosen:
            # (선택) 임시 선택 상태 초기화
            st.session_state[selected_state_key] = set()
            return chosen

        return None

    except Exception as e:
        LOGGER.exception("[CAND][DIALOG] render_candidates_selector_popup failed: %s", e)
        st.error("후보 선택 팝업을 표시하는 중 오류가 발생했습니다.")
        return None






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
                    <div style='padding: 10px 15px; margin-bottom: 10px; background: #262730; 
                                border-left: 3px solid #1f6feb; border-radius: 6px;'>
                        <span style='color: #ffffff; font-size: 14px;'>▪ {model}</span>
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
    Controller that opens a dialog for candidate selection and returns the chosen keys.
    Contract:
      - Returns List[str] after user confirms in dialog.
      - Returns None if not confirmed yet / canceled / closed.
    """
    try:
        # --- 0) Guard
        if not candidates:
            st.info("선택할 후보 모델이 없습니다.")
            return None

        # --- 1) Build stable keys & labels
        model_keys: List[str] = []
        labels: List[str] = []
        for c in candidates:
            mk = (c.get("model_key") or "").strip()
            mn = (c.get("model_name") or mk) or "Unnamed"
            if mk:
                model_keys.append(mk)
                labels.append(f"{mn} · {mk}")

        if not model_keys:
            st.warning("후보 모델에 model_key가 없습니다.")
            return None

        key_suffix = str(abs(hash(tuple(model_keys))))
        ns = f"{session_ns}_{key_suffix}"

        # --- 2) Session keys
        open_flag_key      = f"{ns}_open"
        selected_state_key = f"{ns}_selected"   # temp set of selected keys
        confirmed_key      = f"{ns}_confirmed"  # final chosen list

        # set Default True
        st.session_state[open_flag_key] = True
        # Ensure state container
        if selected_state_key not in st.session_state:
            st.session_state[selected_state_key] = set()

        # --- Optional open button
        if open_button_label:
            if st.button(open_button_label, key=f"{ns}_open_btn", type="primary", use_container_width=True):
                st.session_state[open_flag_key] = True

        # --- 3) Define dialog (modal) once
        @st.dialog(title, width="small")
        def _modal():
            """Dialog UI – cannot return to caller; must set state and rerun."""
            try:
                st.success("질의에 매칭되는 후보모델을 식별하였습니다. 분석을 진행할 모델을 선택하세요. (복수선택 가능)")
                st.write("")

                list_height = 320
                # Checkbox list
                with st.container(height=list_height, border=True):
                    for mk, label in zip(model_keys, labels):
                        ckey = f"{ns}_chk_{mk}"
                        checked = mk in st.session_state[selected_state_key]
                        new_val = st.checkbox(label, key=ckey, value=checked)
                        if new_val:
                            st.session_state[selected_state_key].add(mk)
                        else:
                            st.session_state[selected_state_key].discard(mk)


                if st.button("✓ 확인", key=f"{ns}_ok", type="primary", use_container_width=True):
                    chosen = sorted(list(st.session_state[selected_state_key]))
                    LOGGER.info(
                        "[CAND][CONFIRM] %d selected", len(chosen),
                        extra={"selected": chosen, "count": len(chosen)}
                    )
                    # Save result, close dialog on next run
                    st.session_state[confirmed_key] = chosen
                    st.session_state[open_flag_key] = False
                    st.rerun()

            except Exception as e:
                LOGGER.exception("[CAND][DIALOG] modal failed: %s", e)
                st.error("후보 선택 팝업을 표시하는 중 오류가 발생했습니다.")

        # --- 5) Open dialog if flag set
        if st.session_state.get(open_flag_key, False):
            _modal()

        # --- 6) If confirmed on the previous run, return it now (and clear)
        chosen: Optional[List[str]] = st.session_state.pop(confirmed_key, None)
        if chosen:
            # (선택) 임시 선택 상태 초기화
            st.session_state[selected_state_key] = set()
            return chosen

        return None

    except Exception as e:
        LOGGER.exception("[CAND][DIALOG] render_candidates_selector_popup failed: %s", e)
        st.error("후보 선택 팝업을 표시하는 중 오류가 발생했습니다.")
        return None






