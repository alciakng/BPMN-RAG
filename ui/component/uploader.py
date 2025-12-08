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
from ui.component.common.tree_viewer import render_tree_viewer  # Import tree viewer


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
    Renders category tree viewer, uploader, and graph visualization.
    """
    try:
        LOGGER.info("[UPLOAD][LOADER] Rendering uploader page")

        # Render category tree viewer at the top
        render_category_tree_viewer()

        st.markdown("---")

        # Render uploader and graph
        render_uploader()
        render_graph()
    except Exception as e:
        LOGGER.exception("[UPLOAD][LOADER][ERROR] Render failed: %s", e)
        st.error("Failed to render page. Please check logs.")


def render_category_tree_viewer() -> None:
    """
    Render category tree viewer at the top of uploader page.
    Allows user to select category and predecessor model for upload.
    """
    try:
        st.header("카테고리 선택")

        # Get reader from session
        reader = st.session_state.get("reader")
        if not reader:
            st.warning("Reader가 초기화되지 않았습니다.")
            return

        # Get container ID
        container_id = st.secrets.get("CONTAINER_ID", "default_container")

        # Fetch category tree (categories only)
        try:
            category_tree = reader.fetch_category_tree_only(container_id)
        except Exception as e:
            st.error(f"카테고리 트리 조회 실패: {e}")
            LOGGER.exception("[UPLOADER][TREE] Category tree fetch failed: %s", e)
            return

        if not category_tree:
            st.info("카테고리가 없습니다. 먼저 카테고리를 생성하세요.")
            return

        # 2-column layout: Category selector | Predecessor selector
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. 업로드 대상 카테고리")

            # Render tree viewer
            selected_category = render_tree_viewer(
                tree_data=category_tree,
                key="uploader_category_tree",
                allow_clear=True,
                tree_checkable=False,
                show_search=True,
                max_height=250,
                width_dropdown="100%",
                placeholder="카테고리 선택",
                tree_line=True
            )

            # Log selected_category return value
            LOGGER.info(f"[UPLOADER][TREE] selected_category returned: {selected_category}, type: {type(selected_category)}")

            # Log category_tree structure for debugging
            LOGGER.info(f"[UPLOADER][TREE] category_tree structure: {category_tree}")

            # Store selected category in session state
            if selected_category and len(selected_category) > 0:
                selected_key = selected_category
                LOGGER.info(f"[UPLOADER][TREE] selected_key extracted: {selected_key}")

                category_node = _find_node_in_tree(category_tree, selected_key)
                LOGGER.info(f"[UPLOADER][TREE] category_node found: {category_node}")

                if category_node:
                    st.session_state["upload_category_key"] = selected_key
                    st.session_state["upload_category_name"] = category_node.get("title", selected_key)
                    LOGGER.info(f"[UPLOADER][TREE] Session state updated: key={selected_key}, name={category_node.get('title')}")
                    st.success(f"선택: `{category_node.get('title')}`")
                else:
                    LOGGER.warning(f"[UPLOADER][TREE] category_node is None for selected_key: {selected_key}")
            else:
                st.session_state["upload_category_key"] = None
                st.session_state["upload_category_name"] = None
                LOGGER.info("[UPLOADER][TREE] No category selected")
                st.info("카테고리를 선택하세요.")

        with col2:
            st.markdown("#### 2. 기존 프로세스")

            category_key = st.session_state.get("upload_category_key")

            if category_key:
                # Fetch models under selected category
                try:
                    sibling_models = reader.fetch_models_under_category(category_key)
                    LOGGER.info(f"[UPLOADER][MODELS] Found {len(sibling_models)} models under category {category_key}")
                except Exception as e:
                    st.error(f"모델 조회 실패: {e}")
                    LOGGER.exception("[UPLOADER][MODELS] Sibling models fetch failed: %s", e)
                    sibling_models = []

                if sibling_models:
                    # Build graph data for visualization
                    model_keys = [m.get("model_key") for m in sibling_models]

                    # Create nodes and edges for agraph
                    nodes = []
                    edges = []

                    for model in sibling_models:
                        model_key = model.get("model_key")
                        model_name = model_key

                        # Add node
                        nodes.append({
                            "id": model_key,
                            "label": model_name,
                            "title": model_name,
                            "group": "Model"
                        })

                        # Add edge if NEXT_PROCESS relationship exists
                        next_key = model.get("next_model_key")
                        if next_key:
                            edges.append({
                                "source": model_key,
                                "target": next_key,
                                "label": "NEXT_PROCESS"
                            })

                    LOGGER.info(f"[UPLOADER][GRAPH] Built graph with {len(nodes)} nodes, {len(edges)} edges")

                    # Render graph with agraph
                    from streamlit_agraph import agraph, Node, Edge, Config

                    # Transform to agraph objects
                    from ui.component.agraph import _make_nodes_edges, _agraph_config

                    n_objs, e_objs = _make_nodes_edges(nodes, edges)
                    cfg = _agraph_config(
                        title="Process Graph",
                        height=300,
                        width=600
                    )

                    # Render graph and get selected node
                    selected_node = agraph(nodes=n_objs, edges=e_objs, config=cfg)

                    LOGGER.info(f"[UPLOADER][GRAPH] Selected node: {selected_node}")

                    # Handle node selection
                    if selected_node:
                        # Extract model_key from selected node
                        selected_model_key = selected_node

                        # Find model info
                        selected_model = next(
                            (m for m in sibling_models if m.get("model_key") == selected_model_key),
                            None
                        )

                        if selected_model:
                            predecessor_id = selected_model.get("id")
                            predecessor_name = selected_model.get("name")

                            LOGGER.info(
                                f"[UPLOADER][GRAPH] Setting predecessor: id={predecessor_id}, "
                                f"name={predecessor_name}, model_key={selected_model_key}"
                            )

                            st.session_state["upload_predecessor_key"] = predecessor_id
                            st.session_state["upload_predecessor_name"] = predecessor_name
                            st.success(f"선택된 노드: `{predecessor_name}`")
                        else:
                            LOGGER.warning(f"[UPLOADER][GRAPH] Selected model not found: {selected_model_key}")
                    else:
                        st.session_state["upload_predecessor_key"] = None
                        st.session_state["upload_predecessor_name"] = None
                        st.info("그래프에서 선행 프로세스를 선택하세요")
                else:
                    st.session_state["upload_predecessor_key"] = None
                    st.session_state["upload_predecessor_name"] = None
                    st.info("기존 모델 없음")
            else:
                st.session_state["upload_predecessor_key"] = None
                st.session_state["upload_predecessor_name"] = None
                st.warning("먼저 카테고리를 선택하세요.")

        # Display selected information
        _display_upload_target_info()

    except Exception as e:
        LOGGER.exception("[UPLOADER][TREE][ERROR] Tree viewer rendering failed: %s", e)
        st.error(f"트리 뷰어 렌더링 실패: {e}")


def _find_node_in_tree(tree_data: List[Dict], target_value: str) -> Optional[Dict]:
    """
    Find node in tree data by value (recursive).
    """
    LOGGER.info(f"[UPLOADER][TREE][SEARCH] Searching for target_value: {target_value}")
    LOGGER.info(f"[UPLOADER][TREE][SEARCH] tree_data length: {len(tree_data)}")

    for idx, node in enumerate(tree_data):
        node_value = node.get("value")
        node_title = node.get("title")
        LOGGER.info(f"[UPLOADER][TREE][SEARCH] Checking node[{idx}]: value={node_value}, title={node_title}")

        if node_value == target_value:
            LOGGER.info(f"[UPLOADER][TREE][SEARCH] Match found! Returning node: {node}")
            return node

        children = node.get("children", [])
        if children:
            LOGGER.info(f"[UPLOADER][TREE][SEARCH] Node has {len(children)} children, searching recursively")
            result = _find_node_in_tree(children, target_value)
            if result:
                LOGGER.info(f"[UPLOADER][TREE][SEARCH] Match found in children! Returning: {result}")
                return result

    LOGGER.warning(f"[UPLOADER][TREE][SEARCH] No match found for target_value: {target_value}")
    return None


def _display_upload_target_info() -> None:
    """
    Display upload target information (category and predecessor) in a bordered box.
    """
    try:
        category_key = st.session_state.get("upload_category_key")
        category_name = st.session_state.get("upload_category_name")
        predecessor_key = st.session_state.get("upload_predecessor_key")
        predecessor_name = st.session_state.get("upload_predecessor_name")

        if category_key:
            # Build content
            if predecessor_key:
                content = f"""
                <div style="border: 2px solid #0e76a8; border-radius: 8px; padding: 16px; background-color: #f0f8ff; margin-top: 16px;">
                    <p style="margin: 0; font-size: 15px; color: #333;">
                        <strong>카테고리:</strong> <code>{category_name}</code> 하위 모델로 적재
                    </p>
                    <p style="margin: 8px 0 0 0; font-size: 15px; color: #333;">
                        <strong>선행 프로세스:</strong> <code>{predecessor_name}</code> 의 후행 프로세스로 연결 (NEXT_PROCESS)
                    </p>
                </div>
                """
            else:
                content = f"""
                <div style="border: 2px solid #0e76a8; border-radius: 8px; padding: 16px; background-color: #f0f8ff; margin-top: 16px;">
                    <p style="margin: 0; font-size: 15px; color: #333;">
                        <strong>카테고리:</strong> <code>{category_name}</code> 하위 모델로 적재
                    </p>
                    <p style="margin: 8px 0 0 0; font-size: 15px; color: #666;">
                        선행 프로세스 없음
                    </p>
                </div>
                """

            st.markdown(content, unsafe_allow_html=True)

    except Exception as e:
        LOGGER.exception("[UPLOADER][INFO][ERROR] Display info failed: %s", e)


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

        # Validate category selection
        category_key = st.session_state.get("upload_category_key")
        if not category_key:
            st.error(" 도메인을 선택 후 적재를 진행하세요.")
            return

        # Get predecessor key (optional)
        predecessor_key = st.session_state.get("upload_predecessor_key")

        LOGGER.info(
            "[UPLOAD][SUBMIT] Starting ingestion category=%s predecessor=%s (type=%s)",
            category_key, predecessor_key, type(predecessor_key)
        )

        # Ingest BPMN with category and predecessor
        _process_bpmn_upload(
            bpmn_file=bpmn_file,
            parent_category_key=category_key,
            predecessor_model_key=predecessor_key
        )
        
        # Optional image upload
        if image_file is not None:
            _process_image_upload(image_file)
            
    except Exception as e:
        LOGGER.exception("[UPLOAD][UPLOADER][ERROR] Uploader rendering failed: %s", e)
        st.error("Uploader encountered an error. Please check logs.")


def _process_bpmn_upload(
    bpmn_file,
    parent_category_key: Optional[str] = None,
    predecessor_model_key: Optional[str] = None
) -> None:
    """
    Process and ingest BPMN file into Neo4j.

    Args:
        bpmn_file: Streamlit UploadedFile object
        parent_category_key: Parent category key (for CONTAINS_MODEL relationship)
        predecessor_model_key: Predecessor model key (for NEXT_PROCESS relationship)
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

            # Configure container settings from secrets
            container_settings = ContainerSettings(
                create_container=True,
                container_type=st.secrets.get("CONTAINER_TYPE", "bp"),
                container_id=st.secrets.get("CONTAINER_ID", "bpCntr"),
                container_name=st.secrets.get("CONTAINER_NAME", "BpBpmn")
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
                    container_settings=container_settings,
                    parent_category_key=parent_category_key,
                    predecessor_model_key=predecessor_model_key
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
        LOGGER.info("[UPLOAD][IMAGE] Processing image : %s", image_file.name)
        
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