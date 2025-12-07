# session_store.py
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional
from common.logger import Logger
import streamlit as st

LOGGER = Logger.get_logger("manager.session_store")

class _MemoryBackend:
    """In-memory fallback when Redis is unavailable."""

    def __init__(self):
        self.kv: Dict[str, Any] = {}
        self.lists: Dict[str, List[str]] = {}
        self.sets: Dict[str, set] = {}

    # KV ops
    def get(self, k: str) -> Optional[str]:
        return self.kv.get(k)

    def set(self, k: str, v: str) -> None:
        self.kv[k] = v

    def delete(self, *ks: str) -> None:
        for k in ks:
            self.kv.pop(k, None)
            self.lists.pop(k, None)
            self.sets.pop(k, None)

    # List ops
    def rpush(self, k: str, v: str) -> int:
        lst = self.lists.setdefault(k, [])
        lst.append(v)
        return len(lst)

    def lrange(self, k: str, start: int, end: int) -> List[str]:
        lst = self.lists.get(k, [])
        if end == -1:
            end = len(lst) - 1
        return lst[start : end + 1]

    def ltrim(self, k: str, start: int, end: int) -> None:
        lst = self.lists.get(k, [])
        if end == -1:
            end = len(lst) - 1
        self.lists[k] = lst[start : end + 1]

    # Set ops
    def sadd(self, k: str, *vals: str) -> None:
        st = self.sets.setdefault(k, set())
        for v in vals:
            st.add(v)

    def smembers(self, k: str) -> List[str]:
        return list(self.sets.get(k, set()))

    def srem(self, k: str, *vals: str) -> None:
        st = self.sets.get(k, set())
        for v in vals:
            st.discard(v)


class SessionStore:
    """
    Redis-based session store (with in-memory fallback) to manage:
      - uploaded model key per session
      - selected candidate model keys per session
      - analysis units: analysis_id -> {models, chat history}
      - current analysis id per session
    """

    def __init__(self):
        self.redis = None
        try:
            import redis  # type: ignore
            url = st.secrets.get('REDIS_URL', None)
            if url:
                self.redis = redis.Redis.from_url(url, decode_responses=True)
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
                self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # ping
            self.redis.ping()
            LOGGER.info("[SESSION] Connected to Redis.")
        except Exception as e:
            LOGGER.warning("[SESSION] Redis unavailable, falling back to memory. err=%s", e)
            self.redis = _MemoryBackend()

    # ---------- Key helpers ----------
    @staticmethod
    def _k_uploaded_model(session_id: str) -> str:
        return f"uploaded_model:{session_id}"
    
    @staticmethod
    def _k_uploader_version(session_id: str) -> str:
        """Key for file uploader version counter."""
        return f"uploader_version:{session_id}"

    @staticmethod
    def _k_candidates(session_id: str) -> str:
        return f"selected_candidates:{session_id}"

    @staticmethod
    def _k_current_analysis(session_id: str) -> str:
        return f"current_analysis:{session_id}"

    @staticmethod
    def _k_analysis_models(session_id: str, analysis_id: str) -> str:
        return f"analysis:{session_id}:{analysis_id}:models"

    @staticmethod
    def _k_analysis_history(session_id: str, analysis_id: str) -> str:
        return f"analysis:{session_id}:{analysis_id}:history"

    @staticmethod
    def _k_analysis_index(session_id: str) -> str:
        return f"analysis_index:{session_id}"

    @staticmethod
    def _k_etl_models(session_id: str) -> str:
        """Key for ETL (uploaded via BPMN loader) model keys."""
        return f"etl_models:{session_id}"

    # ---------- Uploaded model ----------
    def save_uploaded_model(self, session_id: str, model_key: str) -> None:
        """Store uploaded model key for the session."""
        try:
            self.redis.set(self._k_uploaded_model(session_id), model_key)

            LOGGER.info("[SESSION] save_uploaded_model session=%s key=%s", session_id, model_key)
        except Exception as e:
            LOGGER.exception("[SESSION][save_uploaded_model][ERROR] %s", e)

    def get_uploaded_model(self, session_id: str) -> Optional[str]:
        """Get uploaded model key."""
        try:
            return self.redis.get(self._k_uploaded_model(session_id))
    
        except Exception as e:
            LOGGER.exception("[SESSION][get_uploaded_model][ERROR] %s", e)
            return None

    # Add after clear_uploaded_model method (around line 123)
    def get_uploader_key(self, session_id: str) -> str:
        """
        Generate dynamic file uploader key for the session.
        Returns a versioned key that changes when reset is triggered.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            str: Uploader key in format "bpmn_uploader_{version}"
            
        Example:
            Initial: "bpmn_uploader_0"
            After reset: "bpmn_uploader_1"
        """
        try:
            version_str = self.redis.get(self._k_uploader_version(session_id))
            version = int(version_str) if version_str else 0
            key = f"bpmn_uploader_{version}"
            
            LOGGER.debug(
                "[SESSION] Generated uploader key",
                extra={
                    "session_id": session_id,
                    "version": version,
                    "key": key
                }
            )
            return key
            
        except Exception as e:
            LOGGER.exception(
                "[SESSION][get_uploader_key][ERROR] %s",
                str(e),
                extra={"session_id": session_id}
            )
            # Fallback to default key on error
            return "bpmn_uploader_0"

    def clear_uploaded_model(self, session_id: str) -> None:
        """Clear selected candidates for the session."""
        try:
            self.redis.delete(self._k_uploaded_model(session_id))

            key = self._k_uploader_version(session_id)
            version_str = self.redis.get(key)
            old_version = int(version_str) if version_str else 0
            new_version = old_version + 1
            
            self.redis.set(key, str(new_version))
            
            LOGGER.info(
                "[SESSION] Incremented uploader version",
                extra={
                    "session_id": session_id,
                    "old_version": old_version,
                    "new_version": new_version
                }
            )

            LOGGER.info("[SESSION] clear_uploaded model session=%s", session_id)
        except Exception as e:
            LOGGER.exception("[SESSION][ clear_uploaded model][ERROR] %s", e)

    # ---------- Selected candidates ----------
    def save_candidates(self, session_id: str, model_keys: List[str]) -> None:
        """Persist selected candidate model keys for overlay panel."""
        try:
            self.redis.set(self._k_candidates(session_id), json.dumps(model_keys, ensure_ascii=False))
            LOGGER.info("[SESSION] save_candidates session=%s", session_id)
        except Exception as e:
            LOGGER.exception("[SESSION][save_candidates][ERROR] %s", e)

    def get_candidates(self, session_id: str) -> List[str]:
        """Get selected candidate model keys."""
        try:
            raw = self.redis.get(self._k_candidates(session_id))
            LOGGER.info("[SESSION] get_candidates session=%s ", session_id)
            return json.loads(raw) if raw else []
        except Exception as e:
            LOGGER.exception("[SESSION][get_candidates][ERROR] %s", e)
            return []

    def clear_candidates(self, session_id: str) -> None:
        """Clear selected candidates for the session."""
        try:
            self.redis.delete(self._k_candidates(session_id))
            LOGGER.info("[SESSION] clear_candidates session=%s", session_id)
        except Exception as e:
            LOGGER.exception("[SESSION][clear_candidates][ERROR] %s", e)

    # ---------- Analysis lifecycle ----------
    def create_analysis(self, session_id: str, model_keys: List[str]) -> str:
        """
        Create a new analysis unit with unique id and bind selected model keys.
        Returns the analysis_id.
        """
        try:
            analysis_id = uuid.uuid4().hex
            self.redis.set(self._k_analysis_models(session_id, analysis_id), json.dumps(model_keys, ensure_ascii=False))
            # index and set as current
            if hasattr(self.redis, "sadd"):
                self.redis.sadd(self._k_analysis_index(session_id), analysis_id)
            self.set_current_analysis(session_id, analysis_id)
            LOGGER.info("[SESSION] create_analysis session=%s analysis=%s models=%s", session_id, analysis_id, model_keys)
            return analysis_id
        except Exception as e:
            LOGGER.exception("[SESSION][create_analysis][ERROR] %s", e)
            # very last resort
            return uuid.uuid4().hex

    def set_current_analysis(self, session_id: str, analysis_id: Optional[str]) -> None:
        """Set (or clear if None) current analysis id for the session."""
        try:
            if analysis_id:
                self.redis.set(self._k_current_analysis(session_id), analysis_id)
            else:
                self.redis.delete(self._k_current_analysis(session_id))
            LOGGER.info("[SESSION] set_current_analysis session=%s analysis=%s", session_id, analysis_id)
        except Exception as e:
            LOGGER.exception("[SESSION][set_current_analysis][ERROR] %s", e)

    def get_current_analysis(self, session_id: str) -> Optional[str]:
        """Get current analysis id for the session."""
        try:
            return self.redis.get(self._k_current_analysis(session_id))
        except Exception as e:
            LOGGER.exception("[SESSION][get_current_analysis][ERROR] %s", e)
            return None

    def get_analysis_models(self, session_id: str, analysis_id: str) -> List[str]:
        """Return model keys bound to analysis id."""
        try:
            raw = self.redis.get(self._k_analysis_models(session_id, analysis_id))
            return json.loads(raw) if raw else []
        except Exception as e:
            LOGGER.exception("[SESSION][get_analysis_models][ERROR] %s", e)
            return []

    def delete_analysis(self, session_id: str, analysis_id: str) -> None:
        """Delete an analysis (models + history) and unindex it."""
        try:
            self.redis.delete(self._k_analysis_models(session_id, analysis_id))
            self.redis.delete(self._k_analysis_history(session_id, analysis_id))
            # unindex
            if hasattr(self.redis, "srem"):
                self.redis.srem(self._k_analysis_index(session_id), analysis_id)
            # if current points to this, clear it
            cur = self.get_current_analysis(session_id)
            if cur == analysis_id:
                self.set_current_analysis(session_id, None)
            LOGGER.info("[SESSION] delete_analysis session=%s analysis=%s", session_id, analysis_id)
        except Exception as e:
            LOGGER.exception("[SESSION][delete_analysis][ERROR] %s", e)

    # ---------- Chat history per analysis ----------
    def append_history(self, session_id: str, analysis_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """
        Append a history message to the analysis.
        Returns new length of the history list.
        """
        try:
            entry = {
                "ts": int(time.time()),
                "role": role,
                "content": content,
            }
            if meta:
                entry["meta"] = meta
            key = self._k_analysis_history(session_id, analysis_id)
            payload = json.dumps(entry, ensure_ascii=False)
            ln = self.redis.rpush(key, payload)
            LOGGER.info("[SESSION] append_history session=%s analysis=%s role=%s", session_id, analysis_id, role)
            # hard cap (optional)
            try:
                max_len = int(os.getenv("SESSION_HISTORY_MAXLEN", "200"))
                if max_len > 0 and hasattr(self.redis, "ltrim"):
                    self.redis.ltrim(key, -max_len, -1)
            except Exception:
                pass
            return ln if isinstance(ln, int) else 0
        except Exception as e:
            LOGGER.exception("[SESSION][append_history][ERROR] %s", e)
            return 0

    def get_history(self, session_id: str, analysis_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return history entries as list of dicts (oldest → newest)."""
        try:
            items = self.redis.lrange(self._k_analysis_history(session_id, analysis_id), 0, -1)
            out: List[Dict[str, Any]] = []
            for s in items:
                try:
                    out.append(json.loads(s))
                except Exception:
                    pass
            if limit and limit > 0:
                out = out[-limit:]
            return out
        except Exception as e:
            LOGGER.exception("[SESSION][get_history][ERROR] %s", e)
            return []

    def clear_history(self, session_id: str, analysis_id: str) -> None:
        """Delete history list for the analysis."""
        try:
            self.redis.delete(self._k_analysis_history(session_id, analysis_id))
            LOGGER.info("[SESSION] clear_history session=%s analysis=%s", session_id, analysis_id)
        except Exception as e:
            LOGGER.exception("[SESSION][clear_history][ERROR] %s", e)

    # ---------- ETL Models (BPMN Loader) ----------
    def add_etl_model(self, session_id: str, model_key: str) -> None:
        """
        Add a model key to the ETL models list.
        Used when a BPMN file is uploaded via the loader page.

        Args:
            session_id: Session identifier
            model_key: Model key to add (typically filename without extension)
        """
        try:
            key = self._k_etl_models(session_id)

            # Get existing models
            models = self.get_etl_models(session_id)

            # Add if not already present
            if model_key not in models:
                models.append(model_key)
                self.redis.set(key, json.dumps(models, ensure_ascii=False))
                LOGGER.info(
                    "[SESSION] add_etl_model session=%s model=%s",
                    session_id, model_key
                )
            else:
                LOGGER.debug(
                    "[SESSION] ETL model already exists session=%s model=%s",
                    session_id, model_key
                )
        except Exception as e:
            LOGGER.exception("[SESSION][add_etl_model][ERROR] %s", e)

    def get_etl_models(self, session_id: str) -> List[str]:
        """
        Get all ETL model keys for the session.

        Args:
            session_id: Session identifier

        Returns:
            List of model keys loaded via BPMN loader
        """
        try:
            raw = self.redis.get(self._k_etl_models(session_id))
            models = json.loads(raw) if raw else []
            LOGGER.debug(
                "[SESSION] get_etl_models session=%s count=%d",
                session_id, len(models)
            )
            return models
        except Exception as e:
            LOGGER.exception("[SESSION][get_etl_models][ERROR] %s", e)
            return []

    def remove_etl_model(self, session_id: str, model_key: str) -> None:
        """
        Remove a specific model key from ETL models list.

        Args:
            session_id: Session identifier
            model_key: Model key to remove
        """
        try:
            models = self.get_etl_models(session_id)

            if model_key in models:
                models.remove(model_key)
                key = self._k_etl_models(session_id)
                self.redis.set(key, json.dumps(models, ensure_ascii=False))
                LOGGER.info(
                    "[SESSION] remove_etl_model session=%s model=%s",
                    session_id, model_key
                )
            else:
                LOGGER.debug(
                    "[SESSION] ETL model not found for removal session=%s model=%s",
                    session_id, model_key
                )
        except Exception as e:
            LOGGER.exception("[SESSION][remove_etl_model][ERROR] %s", e)

    def clear_etl_models(self, session_id: str) -> None:
        """
        Clear all ETL models for the session.

        Args:
            session_id: Session identifier
        """
        try:
            self.redis.delete(self._k_etl_models(session_id))
            LOGGER.info("[SESSION] clear_etl_models session=%s", session_id)
        except Exception as e:
            LOGGER.exception("[SESSION][clear_etl_models][ERROR] %s", e)
