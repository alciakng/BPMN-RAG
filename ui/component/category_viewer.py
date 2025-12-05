# ui/component/category_viewer.py
"""
카테고리 계층 구조 탐색기
- 왼쪽: 카테고리 트리 (카테고리 + 모델)
- 오른쪽: 선택된 카테고리의 모델 그래프 (NEXT_PROCESS 관계 포함)
"""
from __future__ import annotations
import streamlit as st
from typing import Optional, List, Dict, Any
from common.logger import Logger
from ui.component.common.tree_viewer import render_tree_viewer
from ui.component.agraph import _make_nodes_edges, _agraph_config
from streamlit_agraph import agraph

LOGGER = Logger.get_logger("ui.category_viewer")


def render_category_explorer():
    """
    카테고리 탐색기 메인 UI
    """
    try:
        st.subheader("📁 카테고리 탐색기")

        # 세션 컨텍스트
        reader = st.session_state.get("reader")

        if not reader:
            st.error("Reader가 초기화되지 않았습니다.")
            return

        # 컨테이너 ID 가져오기
        container_id = st.secrets.get("CONTAINER_ID", "default_container")

        # 2단 레이아웃: 좌측 트리, 우측 그래프
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📂 카테고리 구조")

            # 카테고리 + 모델 트리 조회
            try:
                tree_data = reader.fetch_category_tree_with_models(container_id)
            except Exception as e:
                st.error(f"카테고리 트리 조회 실패: {e}")
                LOGGER.exception("[CATEGORY_VIEWER] Tree fetch failed: %s", e)
                return

            if not tree_data:
                st.info("카테고리가 없습니다.")
                return

            # 트리 뷰어 렌더링
            selected = render_tree_viewer(
                tree_data=tree_data,
                key="category_explorer_tree",
                allow_clear=True,
                tree_checkable=False,
                show_search=True,
                max_height=600,
                width_dropdown="100%",
                placeholder="카테고리 또는 모델 선택",
                tree_line=True
            )

            # 선택된 항목 표시
            if selected and len(selected) > 0:
                selected_key = selected[0]

                # 선택된 노드 정보 찾기
                selected_node_info = _find_node_in_tree(tree_data, selected_key)

                if selected_node_info:
                    is_category = selected_node_info.get("is_category", False)
                    st.markdown("---")
                    st.markdown("**✅ 선택된 항목:**")

                    if is_category:
                        st.info(f"📁 카테고리: `{selected_node_info.get('title')}`")
                    else:
                        st.success(f"📄 모델: `{selected_node_info.get('title')}`")

        with col2:
            if selected and len(selected) > 0:
                selected_key = selected[0]
                selected_node_info = _find_node_in_tree(tree_data, selected_key)

                if selected_node_info:
                    is_category = selected_node_info.get("is_category", False)

                    if is_category:
                        # 카테고리 선택: 하위 모델들의 NEXT_PROCESS 관계 그래프 표시
                        _render_category_models_graph(selected_key, selected_node_info.get("title"), reader)
                    else:
                        # 모델 선택: 해당 모델의 프로세스 그래프 표시 (4개 탭)
                        _render_single_model_graph_tabs(selected_key, selected_node_info.get("title"), reader)
            else:
                st.info("👈 왼쪽에서 카테고리 또는 모델을 선택하세요.")

    except Exception as e:
        LOGGER.exception("[CATEGORY_VIEWER][ERROR] %s", e)
        st.error(f"카테고리 탐색기 렌더링 중 오류 발생: {e}")


def _find_node_in_tree(tree_data: List[Dict[str, Any]], target_value: str) -> Optional[Dict[str, Any]]:
    """
    트리 데이터에서 특정 value를 가진 노드 찾기 (재귀)
    """
    for node in tree_data:
        if node.get("value") == target_value:
            return node

        # 하위 노드 검색
        children = node.get("children", [])
        if children:
            result = _find_node_in_tree(children, target_value)
            if result:
                return result

    return None


def _render_category_models_graph(category_key: str, category_name: str, reader):
    """
    카테고리 하위 모델들의 NEXT_PROCESS 관계 그래프 표시
    """
    try:
        st.markdown(f"### 📊 카테고리: `{category_name}`")
        st.markdown("#### 하위 모델 프로세스 흐름")

        # 카테고리 하위 모델 조회
        models = reader.fetch_models_under_category(category_key)

        if not models:
            st.info("이 카테고리에는 모델이 없습니다.")
            return

        # 모델 목록 표시
        with st.expander("📋 모델 목록", expanded=True):
            for idx, model in enumerate(models, 1):
                prev_icon = "⬅️" if model.get("prev_model_key") else ""
                next_icon = "➡️" if model.get("next_model_key") else ""

                st.markdown(f"{idx}. {prev_icon} `{model.get('name')}` {next_icon}")

        # NEXT_PROCESS 관계 그래프 표시
        model_keys = [m.get("model_key") for m in models if m.get("model_key")]

        if len(model_keys) > 0:
            st.markdown("#### 🔀 NEXT_PROCESS 관계 그래프")

            graph_data = reader.get_models_with_next_process(model_keys)

            if graph_data.get("edges"):
                _render_graph(graph_data, "NEXT_PROCESS Graph", height=500)
            else:
                st.info("모델 간 NEXT_PROCESS 관계가 없습니다.")

    except Exception as e:
        LOGGER.exception("[CATEGORY_MODELS_GRAPH][ERROR] %s", e)
        st.error(f"그래프 렌더링 실패: {e}")


def _render_single_model_graph_tabs(model_key: str, model_name: str, reader):
    """
    단일 모델의 프로세스 그래프 표시 (4개 탭)
    - Overall Process
    - Message Flow
    - Subprocess
    - Data I/O
    """
    try:
        st.markdown(f"### 📄 모델: `{model_name}`")

        # 4가지 그래프 뷰 생성
        tabs = st.tabs([
            "Overall Process",
            "Message Flow",
            "Subprocess",
            "Data I/O"
        ])

        view_methods = [
            (reader.get_overall_process_graph, "Overall Process Graph"),
            (reader.get_message_exchange_graph, "Message Flow Graph"),
            (reader.get_subprocess_graph, "Subprocess Graph"),
            (reader.get_data_io_graph, "Data I/O Graph")
        ]

        for idx, (tab, (method, title)) in enumerate(zip(tabs, view_methods)):
            with tab:
                try:
                    graph_data = method(model_key)

                    if graph_data.get("nodes"):
                        _render_graph(graph_data, title, height=640 + idx)
                    else:
                        st.info(f"{title} 데이터가 없습니다.")

                except Exception as view_err:
                    LOGGER.exception("[MODEL_GRAPH_TAB][ERROR] Tab %s failed: %s", title, view_err)
                    st.error(f"{title} 렌더링 실패: {view_err}")

    except Exception as e:
        LOGGER.exception("[SINGLE_MODEL_GRAPH][ERROR] %s", e)
        st.error(f"모델 그래프 렌더링 실패: {e}")


def _render_graph(graph_data: Dict[str, Any], title: str, height: int = 640):
    """
    agraph.py의 기능을 활용한 그래프 렌더링
    """
    try:
        nodes_data = graph_data.get("nodes", [])
        edges_data = graph_data.get("edges", [])

        if not nodes_data:
            st.warning("그래프 노드가 없습니다.")
            return

        # agraph.py의 _make_nodes_edges 함수 활용
        n_objs, e_objs = _make_nodes_edges(nodes_data, edges_data)

        if not n_objs:
            st.info("유효한 노드가 없습니다.")
            return

        # agraph.py의 _agraph_config 함수 활용
        config = _agraph_config(title=title, height=height, width=1000)

        # 그래프 렌더링
        agraph(nodes=n_objs, edges=e_objs, config=config)

        LOGGER.debug("[GRAPH] Rendered %s with %d nodes, %d edges", title, len(n_objs), len(e_objs))

    except Exception as e:
        LOGGER.exception("[GRAPH][ERROR] %s", e)
        st.error(f"그래프 렌더링 중 오류 발생: {e}")
