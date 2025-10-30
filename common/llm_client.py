# services/llm_client.py
from __future__ import annotations
import os
import json
from common.logger import Logger
from typing import Dict, Optional, List
from openai import OpenAI

LOGGER = Logger.get_logger("common.llm_client")

class LLMClient:
    """
    - complete(prompt) for text gen
    - query_embed(text) for query vector embeddings (OpenAI if available; safe fallback otherwise)
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
        try:
            if self.api_key:
                self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            LOGGER.exception("[LLM] OpenAI init failed: %s", e)
            self._client = None

    def complete(self, messages: List[Dict[str, str]]) -> str:
        try:
            if self._client:
                resp = self._client.chat.completions.create(
                    model="gpt-4.1-mini",              
                    messages=messages,
                    temperature=0.2,            
                    top_p=1,      
                    max_tokens=2500,
                    presence_penalty=0,
                    frequency_penalty=0,
                    # response_format={"type": "text"},  # 필요시 명시
                )
                text =resp.choices[0].message.content or ""
                LOGGER.info("[LLM] finish_reason=%s out_len=%d", getattr(resp.choices[0], "finish_reason", None), len(text))
                return text
        except Exception as e:
            msg = str(e).lower()
            # Detect 429 & suggested wait sec
            if "rate limit" in msg or "429" in msg:
                import re, time, random
                wait = None
                m = re.search(r"try again in ([0-9.]+)s", str(e), flags=re.I)
                if m:
                    try: wait = float(m.group(1))
                    except Exception: wait = None
                if wait is None: wait = 6.0
                wait += random.uniform(0.0, 0.4)
                LOGGER.warning("[LLM][429] sleep %.2fs then retry", wait)
                time.sleep(wait)

                # degrade: compact prompt more, switch model & shrink tokens
                try:
                    resp = self._client.chat.completions.create(
                        model="gpt-4.1-mini",   # lighter model
                        messages=messages,
                        temperature=0.2,
                        top_p=1,
                        max_tokens=1024,        # smaller again
                    )
                    text = resp.choices[0].message.content or ""
                    LOGGER.info("[LLM][retry-mini] out_len=%d", len(text))
                    return text
                except Exception as e2:
                    LOGGER.exception("[LLM][retry-mini] failed: %s", e2)

            LOGGER.exception("[LLM] complete failed: %s", e)
            return f"{type(e).__name__}: {e}"

    def query_embed(self, text: str) -> List[float]:
        """
        Return embedding vector for the input text.
        Uses OpenAI 'text-embedding-3-small' if available; fallback to zero-vector.
        """
        try:
            if self._client:
                er = self._client.embeddings.create(model="text-embedding-3-small", input=[text])
                vec = er.data[0].embedding
                # Ensure float list
                return [float(x) for x in vec]
            # Fallback: 3072-dim zero vector (shape expected by many indices)
            LOGGER.warning("[LLM] embed fallback; returning zero-vector")
            return [0.0] * 3072
        except Exception as e:
            LOGGER.exception("[LLM] embed failed: %s", e)
            return [0.0] * 3072
