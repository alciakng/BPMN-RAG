# init.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
import streamlit as st


from manager.session_store import SessionStore
from manager.reader import Reader
from agent.query_interpreter import QueryInterpreter
from agent.context_composer import ContextComposer
from agent.graph_query_agent import GraphQueryAgent
from common.llm_client import LLMClient  
from common.settings import Settings

logger = logging.getLogger(__name__)

def init_app() -> Dict[str, Optional[Any]]:
    """
    Initialize services and place them into Streamlit session_state.

    Returns:
        {"session_id": str|None, "agent": GraphQueryAgent|None, "session_store": SessionStore|None}
    """
    try:
        logger.info("[INIT] start")

        # Ensure session id
        if "session_id" not in st.session_state:
            st.session_state.session_id = os.urandom(8).hex()
        session_id = st.session_state.session_id

        if "trigger_generate" not in st.session_state:
            st.session_state["trigger_generate"] = False

        # Session store
        if "session_store" not in st.session_state:
            st.session_state.session_store = SessionStore()
        session_store = st.session_state.session_store

        # Reader (Neo4j)
        if "reader" not in st.session_state:

            uri = st.secrets["NEO4J_URI"]
            user = st.secrets["NEO4J_USERNAME"]
            pwd = st.secrets["NEO4J_PASSWORD"]
            db  = st.secrets["NEO4J_DATABASE"]
        
            settings = Settings()
            settings.set_neo4j_config(uri=uri,user_name=user,password=pwd,database=db)

            st.session_state.reader = Reader(settings.neo4j)
        reader = st.session_state.reader

        # LLM Client
        if "llm_client" not in st.session_state:
            open_api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.llm_client = LLMClient(open_api_key)
        llm = st.session_state.llm_client

        # Interpreter & Composer
        if "interpreter" not in st.session_state:
            st.session_state.interpreter = QueryInterpreter(llm_client=llm, reader=reader, logger=logger)
        interpreter = st.session_state.interpreter

        if "composer" not in st.session_state:
            st.session_state.composer = ContextComposer(reader=reader)
        composer = st.session_state.composer

        # Agent
        if "agent" not in st.session_state:
            st.session_state.agent = GraphQueryAgent(
                llm_client=llm,
                reader=reader,
                context_composer=composer,
                interpreter=interpreter,
            )
        agent = st.session_state.agent

        logger.info("[INIT] done; session_id=%s", session_id)
        return {"session_id": session_id, "agent": agent, "session_store": session_store}
    except Exception as e:
        logger.exception("[INIT][ERROR] %s", e)
        return {"session_id": None, "agent": None, "session_store": None}
