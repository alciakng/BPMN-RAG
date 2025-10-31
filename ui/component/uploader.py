# uploader.py (수정된 부분만)
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
import tempfile
import time
from typing import Optional, Tuple, List, Dict

import streamlit as st

from agent.graph_query_agent import GraphQueryAgent
from manager.session_store import SessionStore
from bpmn2neo.settings import ContainerSettings 
from ui.app.handler import fetch_graph_for_tabs, ingest_and_register_bpmn, handle_image_upload_to_s3
from ui.common.log_viewer import LiveLogPanel
from ui.component.agraph import render_graph_with_selector  # Import new graph module


LOGGER = logging.getLogger(__name__)


def _get_agent() -> GraphQueryAgent:
    """Retrieve GraphQueryAgent from session state."""
    return st.session_state.get("agent")


def _get_session_store() -> SessionStore:
    """Retrieve SessionStore from session state."""
    return st.session_state.get("session_store")


def _get_session_id() -> Optional[str]:
    """Retrieve current session ID from session state."""
    return st.session_state.get("session_id")


def _filename_key(name: str) -> str:
    """
    Extract filename without extension as model/object key.
    
    Args:
        name: Full filename with extension
        
    Returns:
        Filename stem (without extension)
    """
    try:
        base = os.path.basename(name or "").strip()
        stem, _ = os.path.splitext(base)
        return stem
    except Exception as e:
        LOGGER.warning("[UPLOAD][FILENAME_KEY][ERROR] %s", e)
        return "unknown_model"


def render_loader() -> None:
    """
    Main entry point for BPMN uploader page.
    Renders both uploader and graph visualization.
    """
    try:
        LOGGER.info("[UPLOAD][LOADER] Rendering uploader page")
        render_uploader()
        render_graph()
    except Exception as e:
        LOGGER.exception("[UPLOAD][LOADER][ERROR] Render failed: %s", e)
        st.error("Failed to render page. Please check logs.")


def render_graph() -> None:
    """
    Render graph visualization for loaded models.
    Supports single or multiple model selection via slider.
    """
    try:
        # Get loaded model keys from session
        loaded_models = st.session_state.get("loaded_model_keys", [])
        
        if not loaded_models:
            LOGGER.info("[UPLOAD][GRAPH] No models loaded yet")
            return
        
        LOGGER.info(
            "[UPLOAD][GRAPH] Rendering graphs for models: %s",
            loaded_models
        )
        
        # Render with model selector (handles single/multiple models)
        render_graph_with_selector(loaded_models)
        
    except Exception as e:
        LOGGER.exception("[UPLOAD][GRAPH][ERROR] Graph rendering failed: %s", e)
        st.error("Failed to render graphs. Please check logs.")


def render_uploader() -> None:
    """
    Render BPMN + Image uploader interface.
    
    Features:
    - BPMN file upload and ingestion via bpmn2neo
    - Optional image upload to S3
    - Live log streaming during ingestion
    """
    try:
        st.title("BPMN Loader")

        # Custom CSS for tab styling
        st.markdown("""
        <style>
        /* Tabs container */
        div[data-baseweb="tab-list"] {
            font-size: 20px !important;
            height: 60px;
        }
        /* Each tab button */
        button[role="tab"] {
            padding: 12px 24px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            color: white !important;
            background-color: #1c1c1c !important;
            border-radius: 8px !important;
            margin-right: 10px !important;
        }
        /* Selected tab */
        button[aria-selected="true"] {
            background-color: #0e76a8 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.info(".bpmn 파일을 업로드하고 적재를 수행하십시오.(파일명은 특수문자를 제거해주세요.)")

        # File upload columns
        col1, col2 = st.columns([3, 2], vertical_alignment="center")

        with col1:
            bpmn_file = st.file_uploader(
                "Upload BPMN (*.bpmn)",
                type=["bpmn"],
                accept_multiple_files=False,
                key="bpmn_file"
            )
                
        with col2:
            image_file = st.file_uploader(
                "Upload Image (optional: png/jpg/jpeg)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
                key="img_file"
            )

        do_ingest = st.button(
            "적재하기",
            type="primary",
            use_container_width=True
        )

        if not do_ingest:
            return

        # Validate BPMN file
        if bpmn_file is None:
            st.error("Please upload a BPMN file first.")
            return
        
        LOGGER.info("[UPLOAD][SUBMIT] Starting ingestion process")
        
        # Ingest BPMN
        _process_bpmn_upload(bpmn_file)
        
        # Optional image upload
        if image_file is not None:
            _process_image_upload(image_file)
            
    except Exception as e:
        LOGGER.exception("[UPLOAD][UPLOADER][ERROR] Uploader rendering failed: %s", e)
        st.error("Uploader encountered an error. Please check logs.")


def _process_bpmn_upload(bpmn_file) -> None:
    """
    Process and ingest BPMN file into Neo4j.
    
    Args:
        bpmn_file: Streamlit UploadedFile object
    """
    tmp_path = None
    
    try:
        with st.spinner('Ingesting BPMN model...'):
            # Create temporary file
            suffix = Path(bpmn_file.name).suffix or ""
            file_bytes = bpmn_file.read()
            
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                prefix="bpmn_"
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            model_key = _filename_key(bpmn_file.name)
            
            LOGGER.info(
                "[UPLOAD][BPMN] Processing file: name=%s key=%s path=%s",
                bpmn_file.name, model_key, tmp_path
            )

            # Get session components
            agent_obj = _get_agent()
            session_store = _get_session_store()
            session_id = _get_session_id()

            # Configure container settings
            container_settings = ContainerSettings(
                create_container=True,
                container_type='bp',
                container_id='bpCntr',
                container_name='BpBpmn'
            )

            # Create live log panel
            panel = LiveLogPanel(logger=LOGGER)

            def _task():
                """Ingest task wrapped for live logging."""
                return ingest_and_register_bpmn(
                    agent=agent_obj,
                    session_store=session_store,
                    session_id=session_id,
                    file_path=tmp_path,
                    filename=model_key,
                    container_settings=container_settings
                )

            # Run ingestion with live log streaming
            res = panel.run_with_stream(
                task=_task,
                close_on_success=True
            )

            # Check result
            if not res.get("ok"):
                LOGGER.error("[UPLOAD][BPMN] Ingestion failed for: %s", model_key)
                st.error("BPMN ingestion failed. Please check logs below.")
                return
            
            # Success - update session state
            info = res.get("result") or {}
            model_name = info.get('model_name') or model_key
            
            st.success(f"Model ingestion complete: {model_name} (key={model_key})")
            
            LOGGER.info(
                "[UPLOAD][BPMN][SUCCESS] session=%s model_key=%s",
                session_id, model_key
            )
            
            # Add to loaded models list
            if "loaded_model_keys" not in st.session_state:
                st.session_state["loaded_model_keys"] = []
            
            if model_key not in st.session_state["loaded_model_keys"]:
                st.session_state["loaded_model_keys"].append(model_key)
            
            # Rerun to show graph
            st.rerun()
            
    except Exception as e:
        LOGGER.exception("[UPLOAD][BPMN][ERROR] Upload processing failed: %s", e)
        st.error("BPMN upload failed. Please check logs.")
        
    finally:
        # Clean up temporary file
        if tmp_path:
            try:
                os.unlink(tmp_path)
                LOGGER.info("[UPLOAD][BPMN][CLEANUP] Removed temp file: %s", tmp_path)
            except Exception as ce:
                LOGGER.warning("[UPLOAD][BPMN][CLEANUP][WARN] Failed to remove temp file: %s", ce)


def _process_image_upload(image_file) -> None:
    """
    Upload image file to S3.
    
    Args:
        image_file: Streamlit UploadedFile object
    """
    try:
        LOGGER.info("[UPLOAD][IMAGE] Processing image: %s", image_file.name)
        
        obj_key, s3_url = handle_image_upload_to_s3(
            image_bytes=image_file.read(),
            filename=image_file.name
        )
        
        st.info(f"Image upload complete: key={obj_key}")
        st.link_button("View on S3", s3_url, use_container_width=False)
        
        LOGGER.info("[UPLOAD][IMAGE][SUCCESS] S3 key=%s", obj_key)
        
    except Exception as e:
        LOGGER.exception("[UPLOAD][IMAGE][ERROR] Image upload failed: %s", e)
        st.warning("Image upload failed (BPMN ingestion unaffected)")