import streamlit as st
import logging

# Global logger
LOGGER = logging.getLogger(__name__)


def intro():
    """
    Render the introduction page for BPMN AI-Agent application.
    Displays welcome message and feature descriptions.
    """
    try:
        # Page title with custom styling - BPMN in green color
        st.markdown(
            """
            <h1 style='text-align: center; font-size: 3.5rem;'>
                <span style='color: #00D26A;'>BPMN</span> AI-Agent
            </h1>
            """,
            unsafe_allow_html=True
        )
        
        # Add spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Welcome message
        st.markdown(
            """
            <div style='text-align: center; font-size: 1.2rem; color: #A0A0A0; margin-bottom: 3rem;'>
                BPMN AI-Agent에 오신 것을 환영합니다!<br>
                비즈니스 프로세스 모델을 적재하고 분석하세요.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Feature sections as clickable cards with buttons
        col1, col2 = st.columns(2, gap="large")
        
        # BPMN Upload Section Button
        with col1:
            st.markdown(
                """
                <div style='background-color: #1E3A5F; padding: 2rem; border-radius: 10px; height: 100%; '>
                    <h2 style='color: #00D26A; margin-bottom: 1rem; text-align: center;'>BPMN 적재</h2>
                    <p style='font-size: 1.1rem; line-height: 1.8; margin-left: 2rem'>
                        • BPMN 파일 업로드 및 GraphDB(Neo4j) 변환 적재<br>
                        • GraphDB 시각화(OverallProcess • 메시지 플로우 • DATA I/O)<br>
                        • Powered by  <a href='https://github.com/alciakng/bpmn2neo' target='_blank' style='color: #00D26A; text-decoration: none;'>
                                    bpmn2neo
                                     </a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

        # Process Analysis Section Button
        with col2:
            st.markdown(
                """
                <div style='background-color: #1E3A5F; padding: 2rem; border-radius: 10px; height: 100%; '>
                    <h2 style='color: #00D26A; margin-bottom: 1rem;text-align: center;'>프로세스 분석</h2>
                    <p style='font-size: 1.1rem; line-height: 1.8; margin-left: 2rem'>
                        • BPMN GraphDB 기반 프로세스 마이닝 에이전트<br>
                        • 프로세스의 흐름 및 맥락에 기반한 정확한 답변 제공<br>
                        • 사용자 업로드 모델 진단 및 개선점 제안<br>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

        # Footer with GitHub hyperlink
        st.markdown(
            """
            <div style='text-align: center; color: #606060; font-size: 1.1rem; margin-top: 3rem;'>
                Created by JongHwan Kim · 
                <a href='https://github.com/alciakng' target='_blank' style='color: #00D26A; text-decoration: none;'>
                    GitHub
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        LOGGER.info("Intro page rendered successfully")
        
    except Exception as e:
        # Error handling with logging
        LOGGER.error(f"Error rendering intro page: {str(e)}", exc_info=True)
        st.error(
            "인트로 페이지를 불러오는 중 오류가 발생했습니다. "
            "페이지를 새로고침하거나 문제가 지속되면 지원팀에 문의하세요."
        )


def main_board():
    """
    Main board function to handle page navigation.
    Routes to different pages based on sidebar selection or session state.
    """
    try:
        # Initialize session state for menu selection if not exists
        if 'selected_menu' not in st.session_state:
            st.session_state.selected_menu = "Main"
        
        # Get selected menu from sidebar
        selected = sidebar_menu()
        
        # Update session state if sidebar selection changes
        if selected != st.session_state.selected_menu:
            st.session_state.selected_menu = selected
        
        # Route to appropriate page based on session state
        if st.session_state.selected_menu.startswith("Main"):
            intro()
        elif st.session_state.selected_menu.startswith("BPMN 적재"):
            render_loader()
        elif st.session_state.selected_menu.startswith("프로세스 분석"):
            handle_agent_response()
            
        LOGGER.info(f"Navigated to: {st.session_state.selected_menu}")
        
    except Exception as e:
        LOGGER.error(f"Error in main_board navigation: {str(e)}", exc_info=True)
        st.error("네비게이션 오류가 발생했습니다. 다시 시도해주세요.")


def sidebar_menu():
    """
    Render sidebar menu for navigation.
    Returns the selected menu item.
    """
    try:
        with st.sidebar:
            st.title("BPMN AI-Agent")
            
            selected = st.radio(
                "메뉴 선택",
                ["Main", "BPMN 적재", "프로세스 분석"],
                index=["Main", "BPMN 적재", "프로세스 분석"].index(st.session_state.get('selected_menu', 'Main')) if st.session_state.get('selected_menu', 'Main') in ["Main", "BPMN 적재", "프로세스 분석"] else 0,
                label_visibility="collapsed"
            )
            
        return selected
        
    except Exception as e:
        LOGGER.error(f"Error rendering sidebar: {str(e)}", exc_info=True)
        return "Main"  # Default fallback


# Placeholder functions (implement these based on your application logic)
def render_loader():
    """Placeholder for BPMN loader page"""
    st.title("BPMN 적재")
    st.info("BPMN 로더 페이지 - 여기에 업로드 로직을 구현하세요")


def handle_agent_response():
    """Placeholder for process analysis page"""
    st.title("프로세스 분석")
    st.info("프로세스 분석 페이지 - 여기에 에이전트 채팅 로직을 구현하세요")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set page config
    st.set_page_config(
        page_title="BPMN AI-Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Run main board
    main_board()