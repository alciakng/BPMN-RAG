# services/llm_client.py
from __future__ import annotations
import os
import json
from common.logger import Logger
from typing import Dict, Optional, List, Literal
from openai import OpenAI

LOGGER = Logger.get_logger("common.llm_client")

# Supported LLM providers and models
LLMProvider = Literal["openai", "anthropic"]
LLMModel = Literal[
    "gpt-4.1-mini",
    "gpt-5",
    "claude-3-5-sonnet-20241022",    # Legacy Claude 3.5 Sonnet
    "claude-3-7-sonnet-20250219",    # Legacy Claude 3.7 Sonnet
    "claude-sonnet-4-5-20250929",    # Claude 4.5 Sonnet (recommended, balanced)
    "claude-haiku-4-5-20251001",     # Claude 4.5 Haiku (fastest, cheapest)
    "claude-opus-4-5-20251101"       # Claude 4.5 Opus (most capable)
]

class LLMClient:
    """
    - complete(prompt) for text gen
    - query_embed(text) for query vector embeddings (OpenAI if available; safe fallback otherwise)

    Supports multiple LLM providers:
      OpenAI:
        - gpt-4.1-mini (default, fast and cost-effective)
        - gpt-5 (most capable OpenAI model)

      Anthropic Claude 4.5 (Latest):
        - claude-sonnet-4-5-20250929 (recommended: $3/$15 per MTok)
        - claude-haiku-4-5-20251001 (fastest: $1/$5 per MTok)
        - claude-opus-4-5-20251101 (most capable: $5/$25 per MTok)

      Anthropic Claude 3.x (Legacy):
        - claude-3-5-sonnet-20241022
        - claude-3-7-sonnet-20250219
    """
    def __init__(self, openai_api_key: str, anthropic_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self._openai_client = None
        self._anthropic_client = None

        # Initialize OpenAI client
        try:
            if self.openai_api_key:
                self._openai_client = OpenAI(api_key=self.openai_api_key)
                LOGGER.info("[LLM] OpenAI client initialized")
        except Exception as e:
            LOGGER.exception("[LLM] OpenAI init failed: %s", e)
            self._openai_client = None

        # Initialize Anthropic client
        try:
            if self.anthropic_api_key:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                LOGGER.info("[LLM] Anthropic client initialized")
        except Exception as e:
            LOGGER.exception("[LLM] Anthropic init failed: %s", e)
            self._anthropic_client = None

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: LLMModel = "gpt-4.1-mini",
        temperature: float = 0.2,
        max_tokens: int = 5000
    ) -> str:
        """
        Generate completion using specified LLM model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (default: "gpt-4.1-mini")
                OpenAI: "gpt-4.1-mini", "gpt-5"
                Claude 4.5: "claude-sonnet-4-5-20250929" (recommended),
                           "claude-haiku-4-5-20251001" (fastest),
                           "claude-opus-4-5-20251101" (most capable)
                Claude 4 : "claude-sonnet-4-20250514"
                Claude 3.x: "claude-3-7-sonnet-20250219"
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        # Determine provider based on model
        if model.startswith("gpt"):
            return self._complete_openai(messages, model, temperature, max_tokens)
        elif model.startswith("claude"):
            return self._complete_anthropic(messages, model, temperature, max_tokens)
        else:
            LOGGER.error("[LLM] Unsupported model: %s", model)
            return f"Error: Unsupported model {model}"

    def _complete_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using OpenAI models (GPT-4.1, GPT-5)"""
        try:
            if not self._openai_client:
                return "Error: OpenAI client not initialized"

            # GPT-5 requires different parameters
            if model.startswith("gpt-5"):
                # GPT-5: use max_completion_tokens, no temperature
                resp = self._openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=7000,
                )
            else:
                # GPT-4.1 and earlier: use max_tokens and temperature
                resp = self._openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=1,
                    max_tokens=max_tokens,
                    presence_penalty=0,
                    frequency_penalty=0,
                )

            text = resp.choices[0].message.content or ""
            LOGGER.info("[LLM][OpenAI][%s] finish_reason=%s out_len=%d",
                       model, getattr(resp.choices[0], "finish_reason", None), len(text))
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
                LOGGER.warning("[LLM][OpenAI][429] sleep %.2fs then retry", wait)
                time.sleep(wait)

                # Retry with reduced tokens
                try:
                    if model.startswith("gpt-5"):
                        # GPT-5: use max_completion_tokens
                        resp = self._openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_completion_tokens=min(1024, max_tokens),
                        )
                    else:
                        # GPT-4.1 and earlier: use max_tokens and temperature
                        resp = self._openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            top_p=1,
                            max_tokens=min(1024, max_tokens),
                        )
                    text = resp.choices[0].message.content or ""
                    LOGGER.info("[LLM][OpenAI][retry] out_len=%d", len(text))
                    return text
                except Exception as e2:
                    LOGGER.exception("[LLM][OpenAI][retry] failed: %s", e2)

            LOGGER.exception("[LLM][OpenAI] complete failed: %s", e)
            return f"{type(e).__name__}: {e}"

    def _complete_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using Anthropic models (Claude 3.x, 4.5 Sonnet/Haiku/Opus)"""
        try:
            if not self._anthropic_client:
                return "Error: Anthropic client not initialized"

            # Convert messages format for Anthropic API
            # Anthropic expects system message separate from conversation
            system_message = None
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Create request
            request_params = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if system_message:
                request_params["system"] = system_message

            resp = self._anthropic_client.messages.create(**request_params)

            # Extract text from response
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text += block.text

            LOGGER.info("[LLM][Anthropic][%s] stop_reason=%s out_len=%d",
                       model, resp.stop_reason, len(text))
            return text

        except Exception as e:
            msg = str(e).lower()
            # Handle rate limits for Anthropic
            if "rate limit" in msg or "429" in msg or "overloaded" in msg:
                import time, random
                wait = 6.0 + random.uniform(0.0, 0.4)
                LOGGER.warning("[LLM][Anthropic][429] sleep %.2fs then retry", wait)
                time.sleep(wait)

                # Retry once
                try:
                    system_message = None
                    anthropic_messages = []

                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"]
                        else:
                            anthropic_messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })

                    request_params = {
                        "model": model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        "max_tokens": min(1024, max_tokens),
                    }

                    if system_message:
                        request_params["system"] = system_message

                    resp = self._anthropic_client.messages.create(**request_params)
                    text = ""
                    for block in resp.content:
                        if hasattr(block, "text"):
                            text += block.text

                    LOGGER.info("[LLM][Anthropic][retry] out_len=%d", len(text))
                    return text
                except Exception as e2:
                    LOGGER.exception("[LLM][Anthropic][retry] failed: %s", e2)

            LOGGER.exception("[LLM][Anthropic] complete failed: %s", e)
            return f"{type(e).__name__}: {e}"

    def query_embed(self, text: str) -> List[float]:
        """
        Return embedding vector for the input text.
        Uses OpenAI 'text-embedding-3-small' if available; fallback to zero-vector.
        """
        try:
            if self._openai_client:
                er = self._openai_client.embeddings.create(model="text-embedding-3-small", input=[text])
                vec = er.data[0].embedding
                # Ensure float list
                return [float(x) for x in vec]
            # Fallback: 3072-dim zero vector (shape expected by many indices)
            LOGGER.warning("[LLM] embed fallback; returning zero-vector")
            return [0.0] * 3072
        except Exception as e:
            LOGGER.exception("[LLM] embed failed: %s", e)
            return [0.0] * 3072
