# ui/live_log_panel.py
from __future__ import annotations

import io
import sys
import time
import queue
import threading
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable, Iterable, Optional

import streamlit as st


class _QueueLogHandler(logging.Handler):
    """Log handler that pushes formatted records to a queue."""
    def __init__(self, q: queue.Queue, level=logging.INFO, fmt: str | None = None):
        super().__init__(level)
        self.q = q
        self.setFormatter(logging.Formatter(fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            # Never raise from logging path
            pass


class _QueueWriter(io.TextIOBase):
    """Writer to funnel stdout/stderr (print) into the same queue."""
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s: str) -> int:
        try:
            s = str(s or "")
            if s.strip():
                self.q.put(s.rstrip("\n"))
            return len(s)
        except Exception:
            return 0

    def flush(self) -> None:
        pass


class LiveLogPanel:
    """
    Streamlit live log console runner.

    Design:
    - Captures logging + print in a background thread while running a task.
    - Streams lines into a black code box (st.code).
    - On success: closes the console and returns {"ok": True, "result": ...}
      On failure: keeps the console and returns {"ok": False, "error": str, "lines": [...]}

    Rules:
    - Use self.logger for structured logs.
    - Wrap public methods with try/except and log errors.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Use project-wide logger if injected; else create module logger
        self.logger = logger or logging.getLogger(__name__)

    def run_with_stream(
        self,
        task: Callable[[], Any],
        *,
        tapped_loggers: Optional[Iterable[logging.Logger]] = None,
        max_lines_keep: int = 15,
        poll_interval_sec: float = 0.05,
        close_on_success: bool = True,
    ) -> dict:
        """
        Execute `task()` in a thread and stream logs to a black console.

        Returns:
            {"ok": True,  "result": Any} on success  (console closed if close_on_success)
            {"ok": False, "error": str, "lines": list[str]} on failure (console kept)

        Notes:
        - `tapped_loggers` defaults to common project loggers + root.
        - Safe to call from Streamlit callback blocks.
        """
        try:
            # Placeholders
            log_ph = st.empty()
            # Queue + handlers
            q: queue.Queue[str] = queue.Queue()
            qh = _QueueLogHandler(q)

            # Choose which loggers to tap
            default_targets = [
                logging.getLogger(),                   # root
                logging.getLogger("bpmn2neo"),
                logging.getLogger("bpmn2neo.loader"),
                logging.getLogger("bpmn2neo.parser"),
                logging.getLogger("bpmn2neo.builder"),
                logging.getLogger("bpmn2neo.embedder"),
                logging.getLogger("bpmn2neo.context_writer"),
                logging.getLogger("bpmn2neo.orchestrator"),
                logging.getLogger("bpmn2neo.reader"),
                logging.getLogger("neo4j"),
                self.logger,
            ]
            targets = list(tapped_loggers or default_targets)

            for lg in targets:
                try:
                    lg.addHandler(qh)
                    if lg.level > logging.INFO:
                        lg.setLevel(logging.INFO)
                    lg.propagate = True
                except Exception:
                    pass  # never break UI

            result_box: dict[str, Any] = {"ok": False, "result": None, "error": None}

            # Runner
            def _job():
                try:
                    stdout_bak, stderr_bak = sys.stdout, sys.stderr
                    sys.stdout, sys.stderr = _QueueWriter(q), _QueueWriter(q)
                    try:
                        out = task()
                        result_box["ok"] = True
                        result_box["result"] = out
                    finally:
                        sys.stdout, sys.stderr = stdout_bak, stderr_bak
                except Exception as ex:
                    result_box["ok"] = False
                    result_box["error"] = str(ex)
                    self.logger.exception("[LIVELOG][TASK][ERROR] %s", ex)

            th = threading.Thread(target=_job, daemon=True)
            th.start()

            # Pump loop
            lines: list[str] = []
            while th.is_alive() or not q.empty():
                try:
                    msg = q.get(timeout=0.15)
                    lines.append(msg)
                except queue.Empty:
                    pass
                # Render last N lines (console style)
                log_ph.code("\n".join(lines[-max_lines_keep:]), language="log")
                time.sleep(poll_interval_sec)

            # Detach handlers
            for lg in targets:
                try:
                    lg.removeHandler(qh)
                except Exception:
                    pass

            # Success / failure UI handling
            if result_box["ok"]:
                if close_on_success:
                    log_ph.empty()  # close console on success
                return {"ok": True, "result": result_box["result"]}
            else:
                # Keep console open for debugging and propagate error message to caller
                return {
                    "ok": False,
                    "error": result_box["error"] or "Unknown error",
                    "lines": lines[-max_lines_keep:],
                }

        except Exception as e:
            self.logger.exception("[LIVELOG][FATAL] %s", e)
            # Best-effort message to user; leave any existing console as-is
            st.error("실시간 로그 패널 실행 중 오류가 발생했습니다. 로그를 확인하세요.")
            return {"ok": False, "error": str(e), "lines": []}
