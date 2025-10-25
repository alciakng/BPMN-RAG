# Small JSON helpers to avoid extra dependencies
from typing import Any, Optional


def _extract_text(resp: Any) -> Optional[str]:
    try:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            ch = resp.get("choices")
            if isinstance(ch, list) and ch:
                msg = ch[0].get("message", {})
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
            for k in ("content", "text"):
                v = resp.get(k)
                if isinstance(v, str):
                    return v
        if isinstance(resp, list) and resp:
            last = resp[-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
        return None
    except Exception:
        return None

def json_dumps_safe(obj: Any) -> str:
    try:
        import json
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"

def json_loads_safe(s: Optional[str]) -> Any:
    try:
        import json
        if not s:
            return None
        return json.loads(s)
    except Exception:
        return None
    
