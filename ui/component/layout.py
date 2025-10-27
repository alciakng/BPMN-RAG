# ui/layout.py
from __future__ import annotations
import streamlit as st
from typing import List, Optional
from ui.component.intro import intro
from ui.component.uploader import render_loader
from ui.component.chat import handle_agent_response
from common.logger import Logger
from ui.component.panels import render_selected_models_sidebar
from streamlit_option_menu import option_menu


# Module-level structured logger (no class in UI module)
LOGGER = Logger.get_logger("ui.layout")

def main_board():

    selected = sidebar_menu()

    if selected.startswith("Main"):
        intro()
    elif selected.startswith("BPMN 적재"):
        render_loader()
    elif selected.startswith("프로세스 분석"):
        # Then UI-only chat routine (agent calls are inside app.handlers via st.session_state.agent)
        handle_agent_response()
    

def render_app_layout(title_text: str) -> None:
    """
    Build the main page layout:
    - Single wide page (no columns split).
    - Sidebar menu on the left.
    - Plain text title on top of the main area.
    - Top-right overlay (read-only) shows selected model keys (optional).
    """
    try:
        st.set_page_config(page_title=title_text or "BPMN Graph-RAG", layout="wide")

        # Sidebar menu (left)
        main_board()
            
        # Top-right overlay: read-only state of current selections
        # If selected_models is None or empty, the overlay shows 'No selection'.
        render_selected_models_sidebar()
        
    except Exception as e:
        LOGGER.exception("[UI][LAYOUT] render_app_layout failed: %s", e)
        st.error("Layout rendering failed.")

def sidebar_menu():

    with st.sidebar:
        selected = option_menu(
            menu_title='BPMN AI-Agent',
            options=[
                "Main",
                "BPMN 적재",
                "프로세스 분석"
            ],
            icons=["none"] * 5,
            menu_icon=["none"],
            default_index=0,
            styles={
                "menu-title": {
                    "font-size": f"15px",
                    "font-weight": "800",
                    "margin-bottom": "8px",
                    "line-height": "1.2",
                },
                "container": {"padding": "0rem"},
                "nav-link": {"font-size": "15px", "text-align": "left", "font_weight":"bold"},
                "nav-link-selected": {"background-color": "#3C8DBC", "font-weight": "bold", "color": "white"}
            }
        )

        return selected