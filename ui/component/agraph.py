# ui/app/graph.py
"""
Graph visualization utilities for BPMN models.
Provides functions to transform graph data and render with streamlit-agraph.
"""
from __future__ import annotations

import json
import logging
from typing import Tuple, List, Dict, Optional

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from ui.app.handler import fetch_graph_for_tabs


LOGGER = logging.getLogger(__name__)


def _map_size_color(group: str | None, title: str | None) -> Tuple[int, str]:
    """
    Map node size and color based on BPMN hierarchy.
    
    Hierarchy order (largest to smallest):
    BPMNModel > Participant > Process > Lane > Activity > Event > Gateway
    
    Args:
        group: Node group classification
        title: Node title/tooltip text
        
    Returns:
        Tuple of (size, color_hex)
    """
    try:
        t = (title or "").lower()
        g = (group or "").lower()

        def is_kind(k: str) -> bool:
            """Check if node matches a specific kind."""
            k = k.lower()
            return g == k or k in t

        # Map hierarchy to size and color
        if is_kind("bpmnmodel"):
            return 28, "#f6f8f9"  # Largest - light blue
        if is_kind("participant"):
            return 24, "#eb2b05"  # Red
        if is_kind("process"):
            return 20, "#ff7f0e"  # Orange
        if is_kind("lane"):
            return 18, "#7be5fb"  # Cyan
        if is_kind("activity"):
            return 16, "#1804f8"  # Blue
        if is_kind("event"):
            return 14, "#ffff07"  # Yellow
        if is_kind("gateway"):
            return 12, "#038f18"  # Green
        
        # Default for unclassified nodes
        return 14, "#7f7f7f"
        
    except Exception as e:
        LOGGER.warning("[GRAPH][MAP_SIZE_COLOR][ERROR] Failed to map node: %s", e)
        return 14, "#7f7f7f"  # Return safe default


def _make_nodes_edges(
    nodes: List[Dict], 
    edges: List[Dict]
) -> Tuple[List[Node], List[Edge]]:
    """
    Transform raw node/edge dictionaries into streamlit-agraph objects.
    
    Expected node structure: {"id", "label", "title", "group"}
    Expected edge structure: {"source", "target", "label"}
    
    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        
    Returns:
        Tuple of (Node objects list, Edge objects list)
    """
    n_objs: List[Node] = []
    seen: set[str] = set()

    try:
        # Process nodes
        for n in nodes:
            nid = str(n.get("id") or "").strip()
            
            # Skip invalid or duplicate nodes
            if not nid or nid in seen:
                continue
            seen.add(nid)

            # Get size and color based on hierarchy
            size, color = _map_size_color(n.get("group"), n.get("title"))

            n_objs.append(
                Node(
                    id=nid,
                    label=str(n.get("label") or nid),
                    title=str(n.get("title") or ""),  # Tooltip text
                    size=size,
                    color=color,
                    shape="dot",
                    font={'color': "#000000", 'size': 14}  # White text
                )
            )

    except Exception as e:
        LOGGER.exception("[GRAPH][MAKE_NODES][ERROR] Node processing failed: %s", e)

    # Process edges
    e_objs: List[Edge] = []
    try:
        for e in edges:
            src = str(e.get("source") or "").strip()
            tgt = str(e.get("target") or "").strip()
            
            # Skip invalid edges or edges with missing nodes
            if not src or not tgt or src not in seen or tgt not in seen:
                continue
                
            e_objs.append(
                Edge(
                    source=src,
                    target=tgt,
                    label=str(e.get("label") or ""),
                    font={'color': "#000000", 'size': 12}  # Light blue text
                )
            )
            
    except Exception as e:
        LOGGER.exception("[GRAPH][MAKE_EDGES][ERROR] Edge processing failed: %s", e)

    LOGGER.info(
        "[GRAPH][MAKE_NODES_EDGES] Processed %d nodes, %d edges", 
        len(n_objs), len(e_objs)
    )
    return n_objs, e_objs


def _agraph_config(
    title: str, 
    height: int = 640, 
    width: int = 1000
) -> Config:
    """
    Create streamlit-agraph configuration object.
    
    Features:
    - Static graph (no rerun on click)
    - Pan and zoom enabled
    - No physics simulation
    - Directed edges
    
    Args:
        title: Graph title (currently unused, kept for compatibility)
        height: Graph container height in pixels
        width: Graph container width in pixels
        
    Returns:
        Config object for agraph
    """
    try:
        return Config(
            width=width,
            height=height,
            directed=True,
            physics=False,
            staticGraph=False,  # Allow interactions
            panAndZoom=True,    # Enable pan/zoom only
            collapsible=True,
            hierarchical=False,
            linkLength=120
        )
    except Exception as e:
        LOGGER.exception("[GRAPH][CONFIG][ERROR] Config creation failed: %s", e)
        # Return minimal safe config
        return Config(width=width, height=height, directed=True)


def render_graph_for_model(model_key: str) -> None:
    """
    Render graph visualization tabs for a single BPMN model.
    
    Creates 4 tabs:
    1. Overall Process - Complete BPMN structure
    2. Message Flow - Message exchange between participants
    3. Subprocess - Subprocess hierarchy
    4. Data I/O - Data input/output relationships
    
    Args:
        model_key: Unique identifier for the BPMN model
    """
    if not model_key:
        LOGGER.warning("[GRAPH][RENDER] No model key provided")
        st.warning("No model selected for graph visualization.")
        return

    try:
        # Fetch graph data for all views
        LOGGER.info("[GRAPH][RENDER] Fetching graphs for model_key=%s", model_key)
        graphs = fetch_graph_for_tabs(model_key=model_key)

        # Log complete graph structure for debugging
        try:
            LOGGER.info(
                "[GRAPH][RENDER][DATA_DUMP] model_key=%s graphs=%s",
                model_key,
                json.dumps(graphs, ensure_ascii=False)
            )
        except Exception as log_err:
            LOGGER.warning("[GRAPH][RENDER][LOG_ERROR] JSON dump failed: %s", log_err)

    except Exception as e:
        LOGGER.exception("[GRAPH][RENDER][ERROR] Failed to fetch graphs: %s", e)
        st.error(f"Failed to load graph data for model: {model_key}")
        return

    # Create visualization tabs
    try:
        tabs = st.tabs([
            "Overall Process",
            "Message Flow",
            "Subprocess",
            "Data I/O"
        ])

        view_specs = [
            ("overall", "Overall Process Graph"),
            ("message", "Message Flow Graph"),
            ("subprocess", "Subprocess Graph"),
            ("dataio", "Data I/O Graph"),
        ]

        # Render all tabs immediately (not lazy)
        for idx, (tab, (view_key, title)) in enumerate(zip(tabs, view_specs)):
            with tab:
                try:
                    _render_single_graph_view(
                        graphs=graphs,
                        view_key=view_key,
                        title=title,
                        height_offset=idx
                    )
                except Exception as tab_err:
                    LOGGER.exception(
                        "[GRAPH][RENDER][TAB_ERROR] Tab '%s' failed: %s",
                        view_key, tab_err
                    )
                    st.error(f"Failed to render {title}")

    except Exception as e:
        LOGGER.exception("[GRAPH][RENDER][ERROR] Tab creation failed: %s", e)
        st.error("Failed to create graph visualization tabs")


def _render_single_graph_view(
    graphs: Dict,
    view_key: str,
    title: str,
    height_offset: int = 0
) -> None:
    """
    Render a single graph view within a tab.
    
    Args:
        graphs: Dictionary containing all graph views
        view_key: Key to access specific view data
        title: Display title for the graph
        height_offset: Offset for varying heights (prevents caching issues)
    """
    try:
        # Extract data for this view
        ds = graphs.get(view_key) or {}
        nodes = ds.get("nodes") or []
        edges = ds.get("edges") or []

        LOGGER.info(
            "[GRAPH][VIEW] Rendering view=%s nodes=%d edges=%d",
            view_key, len(nodes), len(edges)
        )

        # Check if data exists
        if not nodes and not edges:
            st.info(f"No graph data available for {title}")
            return

        # Transform to agraph objects
        n_objs, e_objs = _make_nodes_edges(nodes, edges)

        if not n_objs:
            st.info(f"No valid nodes to display for {title}")
            return

        # Create config with unique height to prevent caching
        cfg = _agraph_config(
            title=title,
            height=640 + height_offset,
            width=1000
        )

        # Render graph
        agraph(nodes=n_objs, edges=e_objs, config=cfg)
        
        LOGGER.info("[GRAPH][VIEW] Successfully rendered view=%s", view_key)

    except Exception as e:
        LOGGER.exception("[GRAPH][VIEW][ERROR] View rendering failed: %s", e)
        st.error(f"Failed to render graph view: {title}")

def render_graph_with_selector(model_keys: List[str]) -> None:
    """
    Render graph visualization with model selector.
    
    - If single model: Display directly
    - If multiple models: Show radio buttons to select model
    
    Args:
        model_keys: List of model keys to visualize
    """
    if not model_keys:
        LOGGER.warning("[GRAPH][SELECTOR] Empty model_keys list")
        return

    try:
        # Single model - render directly
        if len(model_keys) == 1:
            LOGGER.info("[GRAPH][SELECTOR] Rendering single model: %s", model_keys[0])
            st.subheader(f"Model: {model_keys[0]}")
            render_graph_for_model(model_keys[0])
            return

        # Multiple models - show radio button selector
        LOGGER.info(
            "[GRAPH][SELECTOR] Multiple models detected: count=%d",
            len(model_keys)
        )
        
        st.subheader("Select Model to Visualize")
        
        
        # Create radio buttons for model selection
        selected_model = st.radio(
            label="Choose a model",
            options=model_keys,
            index=0,  # Default to first model
            key="model_selector_radio",
            horizontal=False  # Vertical layout (set True for horizontal)
        )
        
        LOGGER.info(
            "[GRAPH][SELECTOR] User selected model: %s",
            selected_model
        )
        
        # Display selected model info
        selected_idx = model_keys.index(selected_model)
        st.info(f"**Selected Model:** {selected_model} ({selected_idx + 1}/{len(model_keys)})")
        
        # Render graph for selected model
        render_graph_for_model(selected_model)

    except Exception as e:
        LOGGER.exception("[GRAPH][SELECTOR][ERROR] Model selector failed: %s", e)
        st.error("Failed to render model selector. Please check logs.")