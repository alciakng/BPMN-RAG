import mimetypes
import os
import re
from typing import List, Optional
import boto3
import streamlit as st
import streamlit.components.v1 as components
from manager.session_store import SessionStore
from common.logger import Logger

LOGGER = Logger.get_logger("ui.common.utils")

def ensure_ui_styles(ns: str = "ku") -> None:
    """Inject once: pill-style radio in sidebar + title typography."""
    try:
        flag = f"__{ns}_styles_injected__"
        if st.session_state.get(flag):
            return
        st.session_state[flag] = True

        st.markdown("""
<style>
/* ----- Sidebar pill-style radio (works across recent Streamlit builds) ----- */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex !important;
  flex-direction: column !important;
  gap: 8px !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
  border: 1px solid rgba(139, 0, 22, 0.35);  /* crimson outline */
  border-radius: 12px;
  padding: 10px 12px;
  cursor: pointer;
  transition: all .12s ease-in-out;
  background: rgba(139,0,22,0.06);
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
  filter: brightness(1.03);
}

/* Selected state (robust selectors) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"][checked] + div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"]:checked + div {
  background: rgba(139,0,22,0.16);
  border-radius: 10px;
  border: 1px solid rgba(139, 0, 22, 0.55);
}

/* Make radio mark invisible → looks like button */
section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
  display: none !important;
}

/* Title typography */
.ku-title {
  line-height: 1.15;
  font-weight: 800;
  letter-spacing: -0.01em;
  margin: 6px 0 14px 0;
}
.ku-title .accent {
  color: var(--ku-accent, #8B0016); /* fallback crimson */
}
</style>
        """, unsafe_allow_html=True)

    except Exception as e:
        LOGGER.exception("[UI][STYLE] inject failed: %s", e)

def scroll_to_bottom(delay_ms: int = 0) -> None:
    """Force-scroll to bottom; includes short repeated attempts after layout settles."""
    try:
        invoke = f"setTimeout(run, {delay_ms});" if delay_ms > 0 else "run();"
        js = f"""
<script>
(function() {{
  try {{
    const run = () => {{
      const doc = window.parent.document;
      const el = doc.scrollingElement || doc.documentElement || doc.body;
      el.scrollTop = el.scrollHeight;
    }};
    {invoke}
    // Repeat a few times to resist late reflows/focus restore
    let n = 0;
    const id = setInterval(() => {{
      run();
      if (++n > 5) clearInterval(id);
    }}, 60);
  }} catch (e) {{}}
}})();
</script>
"""
        components.html(js, height=0)
    except Exception as e:
        LOGGER.exception("[UI] scroll_to_bottom failed: %s", e)


def _collect_model_keys(session_store: SessionStore, session_id: str) -> List[str]:
    """Collect all model keys from session store."""
    keys = []
    
    # Get uploaded model (str)
    uploaded = session_store.get_uploaded_model(session_id)
    if uploaded:
        keys.append(uploaded)
    
    # Get candidates (List[str])
    candidates = session_store.get_candidates(session_id) or []
    keys.extend(candidates)
    
    # Remove duplicates, keep order
    return list(dict.fromkeys(keys))
