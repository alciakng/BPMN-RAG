# query_interpreter.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from common.logger import Logger
from common.llm_client import LLMClient
from manager.reader import Reader
from manager.util import _extract_text, json_dumps_safe

NodeHit = Dict[str, Any]
CandidateModel = Dict[str, Any]

LOGGER = Logger.get_logger("agent.query_interpreter")

class QueryInterpreter:
    """
    End-to-end interpreter with a single integrated retrieval path:
    - reader.search_hybrid_candidates(user_query, qemb, keywords, limit)
    - Re-rank with Score = wc*cos + wb*bm25_norm + wd*d_io_norm
    - Group to model-level top-K; return top-N models
    - Generate intent-aware prompt message (no label returned)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        reader: Reader,
        logger: Optional[logging.Logger] = None,
        top_k_nodes_per_model: int = 10,
        top_n_models: int = 5,
        wc: float = 0.45,
        wb: float = 0.40,
        wd: float = 0.15,
        io_cap: int = 5,
        bm25_norm_eps: float = 1e-6,
        hybrid_limit: int = 200,
    ):
        self.llm = llm_client
        self.reader = reader
        self.k = max(1, int(top_k_nodes_per_model))
        self.n = max(1, int(top_n_models))
        self.wc = float(wc)
        self.wb = float(wb)
        self.wd = float(wd)
        self.io_cap = max(1, int(io_cap))
        self.bm25_norm_eps = float(bm25_norm_eps)
        self.hybrid_limit = int(hybrid_limit)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def interpret(self, user_query: str, uploaded_model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns:
        {
          "candidates": [{model_key, model_name, top_nodes[...], score_total}, ...],
          "prompt_message": str
        }
        """
        try:
            LOGGER.info("[01.PARSE] user_query len=%d uploaded=%s", len(user_query or ""), bool(uploaded_model_key))

            # 1) Embed query (no getattr fallbacks; just use llm_client.embed)
            qemb = self.llm.query_embed(user_query)

            # 2) Integrated retrieval
            rows = self.reader.search_candidates(user_query=user_query, qemb=qemb, limit=self.hybrid_limit)
            LOGGER.info("[02.RETR] candidates_rows=%d", len(rows))

            if not rows:
                return {"candidates": [], "prompt_message": self._fallback_prompt(uploaded_model_key)}

            # 3) Re-rank
            hits = self._rerank(rows)
            LOGGER.info("[03.SCORE] hits=%d", len(hits))

            # 4) Group by model and keep top-K
            candidates = self._aggregate_candidates(hits)
            LOGGER.info("[04.GROUP] models=%d", len(candidates))

            # 5) Sort by score_total and slice top-N
            top_models = sorted(candidates, key=lambda c: c["score_total"], reverse=True)[: self.n]
            LOGGER.info("[05.SELECT] topN=%d", len(top_models))

            # 6) Prompt message
            prompt_message = self._generate_prompt_message(user_query, uploaded_model_key, top_models)
            LOGGER.info("[06.PROMPT] len=%s", prompt_message)

            return {"candidates": top_models, "prompt_message": prompt_message}
        except Exception as e:
            LOGGER.exception("[INTERPRET][ERROR] %s", e)
            return {
                "candidates": [],
                "prompt_message": self._fallback_prompt(uploaded_model_key),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rerank(self, rows: List[Dict[str, Any]]) -> List[NodeHit]:
        """
        Score = cosine
        """
        try:
            
            out: List[NodeHit] = []
            for r in rows:
                cos_sim = float(r.get("cos_sim") or 0.0)

                hit: NodeHit = {
                "node_id":       r.get("node_id"),
                "node_label":    r.get("node_label"),
                "node_name":     r.get("node_name"), 
                "node_context":  r.get("node_context"),
                "lane_id":       r.get("lane_id"),
                "lane_name":     r.get("lane_name"),
                "lane_context":    r.get("lane_context"),
                "process_id":    r.get("process_id"),
                "process_name":  r.get("process_name"),
                "process_context":r.get("process_context"),
                "part_id":    r.get("part_id"),
                "part_name":  r.get("part_name"),
                "part_context":r.get("part_context"),
                "model_key":  r.get("model_key") or "",
                "model_name": r.get("model_name"),
                "model_context": r.get("model_context"),
                "score":      float(cos_sim),
                }
                out.append(hit)

                LOGGER.info(
                    "[03.SCORE][ADD] %s",
                    json.dumps(
                        {
                            "node":  {"id": hit["node_id"], "label": hit["node_label"], "name": hit["node_name"]},
                            "lane":  {"id": hit["lane_id"], "name": hit["lane_name"]},
                            "proc":  {"id": hit["process_id"], "name": hit["process_name"]},
                            "part":  {"id": hit["part_id"], "name": hit["part_name"]},
                            "model": {"key": hit["model_key"], "name": hit["model_name"]},
                            "score": round(hit["score"], 6)
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )

            out.sort(key=lambda x: x["score"], reverse=True)
            return out
        except Exception as e:
            LOGGER.exception("[03.SCORE][ERROR] %s", e)
            return []

    def _aggregate_candidates(self, hits: List[NodeHit]) -> List[CandidateModel]:
        """Group by model_key and keep top-K per model."""
        try:
            by_model: Dict[str, CandidateModel] = {}
            for h in hits:
                mk = h.get("model_key")
                if not mk:
                    continue
                slot = by_model.setdefault(mk, {"model_key": mk, "model_name": h.get("model_name"), "top_nodes": []})
                slot["top_nodes"].append(h)

            out: List[CandidateModel] = []
            for mk, cm in by_model.items():
                topk = sorted(cm["top_nodes"], key=lambda x: x["score"], reverse=True)[: self.k]
                total = sum(float(x["score"]) for x in topk)
                out.append({"model_key": mk, "model_name": cm.get("model_name"), "top_nodes": topk, "score_total": total})
            return out
        except Exception as e:
            LOGGER.exception("[04.GROUP][ERROR] %s", e)
            return []

    def _generate_prompt_message(self, user_query: str, uploaded_model_key: Optional[str], snap: list[dict]) -> str:
        """
        Return a compact RULE-STRING built from deterministic templates.
        Stage-1 LLM is used ONLY to infer intent flags; it must NOT write free-form prompts.
        """
        log = getattr(self, "logger", LOGGER)
        try:
            # 1) Infer intent flags with a tiny model (no sample answers).
            #    Output MUST be a short JSON: {"want_comparison": bool}
            want_comparison = False
            if hasattr(self.llm, "complete"):
                sys = (
                    "Classify if the user likely requests a comparison.\n"
                    "Return ONLY compact JSON like: {\"want_comparison\": true}\n"
                    "Comparison cues: compare, versus, vs, difference, trade-off, better than, 대조, 비교."
                )
                payload = {
                    "user_query": user_query,
                    "uploaded_model_present": bool(uploaded_model_key),
                    "model_count": sum(1 for c in (snap or []) if isinstance(c, dict))
                }
                raw = self.llm.complete([
                    {"role": "system", "content": sys},
                    {"role": "user", "content": json_dumps_safe(payload)}
                ])
                txt = _extract_text(raw) or ""
                try:
                    want_comparison = bool(json.loads(txt).get("want_comparison"))
                except Exception:
                    want_comparison = False

            # 2) Build RULES deterministically (no LLM writing the prompt).
            mcnt = sum(1 for c in (snap or []) if isinstance(c, dict))
            has_upload = bool(uploaded_model_key)

            rules: list[str] = []
            # common scaffolding
            rules += [
                "Create concise sections; avoid repetition.",
                "Ground every claim strictly in payload_blocks and chat_history; if missing, say it is missing.",
                "Cover Data I/O, Message Flows, and cross-lane handoffs when relevant.",
                "Include a 'Risk & Recommendations' section with actionable mitigations when relevant."
            ]
            if has_upload:
                # uploaded vs selected models focus
                rules += [
                    "Prioritize head-to-head comparison between the uploaded model and each selected model.",
                    "Add migration/alignment recommendations between uploaded and selected models."
                ]
            if mcnt >= 2 and (want_comparison or has_upload):
                # comparison shape
                rules += [
                    "For each model, add a top-level section.",
                    "Add a final comparison section: commonalities, differences, risks, recommendations."
                ]
            elif mcnt >= 2:
                # multi-model info synthesis (no head-to-head)
                rules += [
                    "For each model, add a top-level section.",
                    "Provide a brief synthesis section across models (no head-to-head claims)."
                ]
            else:
                # single model
                rules += ["Write for a single-model analysis."]

            # language is NOT set here (you add in answer_with_selected sys_msg)
            prompt_str = "\n".join(f"- {r}" for r in rules)
            return prompt_str.strip()
        except Exception as e:
            log.exception("[06.PROMPT][ERROR] %s", e)
            # safe fallback rules
            return (
                "- Create concise sections; avoid repetition.\n"
                "- Ground only in payload_blocks and chat_history; state if info is missing.\n"
                "- Cover Data I/O, Message Flows, cross-lane handoffs when relevant.\n"
                "- Include 'Risk & Recommendations' when relevant.\n"
                "- Add per-model sections; if uploaded model exists, compare uploaded vs selected models."
            )



    def _fallback_prompt(self, uploaded_model_key: Optional[str]) -> str:
        if uploaded_model_key:
            return ("Write a structured comparison answer by model sections; "
                    "include commonalities, differences, risks, and recommendations.")
        return ("Write a structured answer by model sections if multiple models; "
                "explain end-to-end flows, lane handoffs, message flows, and data I/O.")

   
