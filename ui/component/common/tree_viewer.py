# ui/component/common/tree_viewer.py
"""
st_ant_tree 기반 공통 트리 뷰어

Features:
- 카테고리 계층 구조 표시
- 후보 모델 선택 (multiple 옵션)
- 클릭/체크박스 이벤트 핸들링
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import streamlit as st
from common.logger import Logger

LOGGER = Logger.get_logger("ui.tree_viewer")


def render_tree_viewer(
    tree_data: List[Dict[str, Any]],
    multiple: bool = False,
    key: str = "tree_viewer",
    default_value: Optional[List[str]] = None,
    allow_clear: bool = True,
    tree_checkable: bool = False,
    show_search: bool = False,
    max_height: int = 400,
    width_dropdown: str = "100%",
    placeholder: str = "항목을 선택하세요",
    tree_line: bool = True,
) -> Optional[List[str]]:
    """
    st_ant_tree 기반 공통 트리 뷰어

    Args:
        tree_data: st_ant_tree 형식의 트리 데이터
            [
                {
                    "value": "node_key",  # 노드 식별자 (modelKey)
                    "title": "Node Name",  # 표시 이름
                    "children": [...]  # 하위 노드 (옵션)
                },
                ...
            ]
        multiple: 다중 선택 허용 여부
        key: Streamlit 위젯 키 (고유해야 함)
        default_value: 기본 선택 값 (value 리스트)
        allow_clear: 선택 해제 허용 여부
        tree_checkable: 체크박스 표시 여부
        show_search: 검색 기능 활성화 여부
        max_height: 드롭다운 최대 높이 (px)
        width_dropdown: 드롭다운 너비 (예: "90%", "500px")
        placeholder: 플레이스홀더 텍스트
        tree_line: 트리 라인 표시 여부

    Returns:
        선택된 노드의 value 리스트 (단일 선택 시에도 리스트 형태)

    Example:
        >>> tree_data = [
        ...     {
        ...         "value": "cat_1",
        ...         "title": "카테고리 A",
        ...         "children": [
        ...             {"value": "model_1", "title": "모델 X"}
        ...         ]
        ...     }
        ... ]
        >>> selected = render_tree_viewer(tree_data, multiple=True, tree_checkable=True)
    """
    try:
        # st_ant_tree import (lazy import for optional dependency)
        try:
            from st_ant_tree import st_ant_tree
        except ImportError:
            LOGGER.error("[TREE] st_ant_tree not installed")
            st.error("st_ant_tree 라이브러리가 설치되지 않았습니다. `pip install st-ant-tree`를 실행하세요.")
            return None

        LOGGER.info(
            "[TREE] Rendering tree",
            extra={
                "key": key,
                "multiple": multiple,
                "tree_checkable": tree_checkable,
                "tree_nodes": len(tree_data)
            }
        )

        # st_ant_tree 호출 (정확한 파라미터 사용)
        selected = st_ant_tree(
            treeData=tree_data,
            defaultValue=default_value or [],
            allowClear=allow_clear,
            treeCheckable=tree_checkable,
            showSearch=show_search,
            max_height=max_height,
            width_dropdown=width_dropdown,
            placeholder=placeholder,
            treeLine=tree_line,
            key=key
        )

        LOGGER.debug(
            "[TREE] Selection result",
            extra={
                "key": key,
                "selected": selected,
                "selected_count": len(selected) if selected else 0
            }
        )

        return selected

    except Exception as e:
        LOGGER.exception("[TREE][ERROR] Tree rendering failed: %s", e)
        st.error(f"트리 렌더링 중 오류 발생: {e}")
        return None
