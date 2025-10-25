# context_composer.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set
import streamlit as st
from common.logger import Logger
from manager.reader import Reader


LOGGER = Logger.get_logger("agent.context_composer")

class ContextComposer:
    """
    Build per-model LLM payloads using reader (direct calls, no getattr/safe_call).
    """

    def __init__(self, reader: Reader):
        self.reader = reader

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def build_llm_payload(self, uploaded_model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Build LLM payload blocks from:
            A) Selected models from session (List[str] of model keys)
            B) Optional uploaded model key
        """
        try:
            # --- A) Selected models from session: now strictly List[str] of model keys ---
            model_blocks: List[Dict[str, Any]] = []
            try:
                session_store = st.session_state.get("session_store")
                session_id = st.session_state.get("session_id")
                selected_keys: List[str] = session_store.get_candidates(session_id) if (session_store and session_id) else []
            except Exception as e:
                LOGGER.exception("[CTX][FETCH] failed to load selected model keys: %s", e)
                selected_keys = []

            # Ensure list for safety; skip invalid entries
            if not isinstance(selected_keys, list):
                LOGGER.info("[CTX][FETCH] selected_keys is not a list; using empty list")
                selected_keys = []

            for mk in selected_keys:
                try:
                    # Accept only non-empty strings
                    if not isinstance(mk, str) or not mk.strip():
                        LOGGER.info("[CTX][FETCH] skip invalid model_key=%s", mk)
                        continue

                    LOGGER.info("[CTX][FETCH] fetching model context model_key=%s", mk)
                    block = self.reader.fetch_model_context(mk)
                    if block:
                        model_blocks.append(block)
                        LOGGER.info("[CTX][FETCH] appended model context model_key=%s", mk)
                    else:
                        LOGGER.info("[CTX][FETCH] empty model context model_key=%s", mk)
                except Exception as e:
                    LOGGER.exception("[CTX][FETCH][ERROR] model_key=%s error=%s", mk, e)

            # --- B) Uploaded model context (optional) ---
            upload_block: Optional[Dict[str, Any]] = None
            if uploaded_model_key:
                try:
                    LOGGER.info("[CTX][FETCH] fetching upload model context model_key=%s", uploaded_model_key)
                    upload_block = self.reader.fetch_model_context(uploaded_model_key)
                except Exception as e:
                    LOGGER.exception("[CTX][UPLOAD][ERROR] %s", e)
                    upload_block = None

            result = {"model_context": model_blocks, "upload_model_context": upload_block}
            LOGGER.info("[CTX][DONE] model_blocks=%d", len(model_blocks))
            return result

        except Exception as e:
            LOGGER.exception("[CTX][ERROR] %s", e)
            return {"model_context": [], "upload_model_context": None}

