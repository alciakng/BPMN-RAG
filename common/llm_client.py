# services/llm_client.py
from __future__ import annotations
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
                    model="gpt-4o",              
                    messages=messages,
                    temperature=0.2,            
                    top_p=1,      
                    max_tokens=3000,
                    presence_penalty=0,
                    frequency_penalty=0,
                    # response_format={"type": "text"},  # 필요시 명시
                )
                text =resp.choices[0].message.content or ""
                LOGGER.info("[LLM] finish_reason=%s out_len=%d", getattr(resp.choices[0], "finish_reason", None), len(text))
                return text
        except Exception as e:
            LOGGER.exception("[LLM] complete failed: %s", e)
            err_full = f"{type(e).__name__}: {e}"
            return err_full

    def query_embed(self, text: str) -> List[float]:
        """
        Return embedding vector for the input text.
        Uses OpenAI 'text-embedding-3-small' if available; fallback to zero-vector.
        """
        try:
            if self._client:
                er = self._client.embeddings.create(model="text-embedding-3-large", input=[text])
                vec = er.data[0].embedding
                # Ensure float list
                return [float(x) for x in vec]
            # Fallback: 3072-dim zero vector (shape expected by many indices)
            LOGGER.warning("[LLM] embed fallback; returning zero-vector")
            return [0.0] * 3072
        except Exception as e:
            LOGGER.exception("[LLM] embed failed: %s", e)
            return [0.0] * 3072
