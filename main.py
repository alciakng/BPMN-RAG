# main.py (발췌)
import streamlit as st
from ui.common.utils import scroll_to_bottom
from ui.app.init import init_app
from ui.component.layout import render_app_layout
from ui.component.chat import handle_agent_response



def main():
    # Initialize the agent (kept in st.session_state.agent)
    init_app()
    # Layout first (selected models may be None initially)
    render_app_layout(title_text="BPMN Graph-RAG")

    if st.session_state.pop("_needs_scroll_after_rerun", False):
        scroll_to_bottom()
    
if __name__ == "__main__":
    main()
