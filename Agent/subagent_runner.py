"""Background sub-agent jobs with bounded concurrency (max 3)."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

try:
    from .background_agent_runner import run_short_agent_loop
except ImportError:
    from Agent.background_agent_runner import run_short_agent_loop

_MAX_CONCURRENT = 3
_SUBAGENT_TOOL_NAMES = frozenset({"spawn_subagent", "get_subagent_result"})

_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_SLOT_SEM = threading.Semaphore(_MAX_CONCURRENT)


def _get_bridge() -> Any:
    try:
        from Interface.tui_bridge import get_bridge
        return get_bridge()
    except Exception:
        return None


def _notify_spawn(token: str, task: str, mode: str, parent_id: str) -> None:
    bridge = _get_bridge()
    if bridge is None:
        return
    if hasattr(bridge, "on_subagent_spawn"):
        try:
            bridge.on_subagent_spawn(token=token, task=task, mode=mode, parent_id=parent_id)
            return
        except Exception:
            pass
    try:
        bridge.on_info(f"🧩 Sub-agent [{token}] ({mode}): {task[:80]}")
    except Exception:
        pass


def _notify_done(token: str, status: str, result: Any = None, error: str = "") -> None:
    bridge = _get_bridge()
    if bridge is None:
        return
    if hasattr(bridge, "on_subagent_done"):
        try:
            bridge.on_subagent_done(token=token, status=status, result=result, error=error)
            return
        except Exception:
            pass
    try:
        if status == "done":
            bridge.on_success(f"Sub-agent [{token}] завершён")
        elif status == "error":
            bridge.on_error(f"Sub-agent [{token}]: {error or 'error'}")
    except Exception:
        pass


def _filter_subagent_tools(tools: List[BaseTool]) -> List[BaseTool]:
    return [
        t for t in tools
        if str(getattr(t, "name", "") or "") not in _SUBAGENT_TOOL_NAMES
    ]


def run_subagent_job(
    task: str,
    mode: str,
    parent_id: str,
    tools: List[BaseTool],
    *,
    project_context: str = "",
    bridge: Any = None,
    max_mini_rounds: int = 12,
) -> Dict[str, Any]:
    """Run one sub-agent synchronously (called from worker thread)."""
    task = str(task or "").strip()
    if not task:
        return {"ok": False, "error": "empty_task"}

    safe_tools = _filter_subagent_tools(tools)
    mode_norm = (mode or "creator").strip().lower()
    if mode_norm in ("mini", "langgraph", "fast"):
        return _run_mini_job(task, safe_tools, max_mini_rounds)
    return _run_creator_job(task, safe_tools, project_context, parent_id, bridge)


def _run_mini_job(
    task: str,
    tools: List[BaseTool],
    max_rounds: int,
) -> Dict[str, Any]:
    try:
        from Agent.tool_registry import build_tool_map, bind_tools_safe
        from Agent.llm_provider import get_llm
    except ImportError:
        from tool_registry import build_tool_map, bind_tools_safe
        from llm_provider import get_llm

    tmap = build_tool_map(tools)
    llm, _profile, mname = get_llm("fast")
    llm_wt = bind_tools_safe(llm, mname, tools)
    summary = run_short_agent_loop(task, tmap, llm_wt, max_rounds=max(1, int(max_rounds)))
    return {"ok": True, "mode": "mini", "summary": summary}


def _run_creator_job(
    task: str,
    tools: List[BaseTool],
    project_context: str,
    parent_id: str,
    bridge: Any,
) -> Dict[str, Any]:
    try:
        from Agent.creator_mode import run_creator_mode
        from Agent.creator_summary import format_creator_summary_text
    except ImportError:
        from creator_mode import run_creator_mode
        from creator_summary import format_creator_summary_text

    b = bridge or _get_bridge()
    try:
        if b is not None:
            b.on_info(f"🧩 Sub-agent (Creator): {task[:80]}")
    except Exception:
        pass
    try:
        res = run_creator_mode(
            task=task,
            tools=tools,
            project_context=project_context,
            depth=1,
            parent_worker_id=parent_id or "subagent",
        )
        return {
            "ok": True,
            "mode": "creator",
            "summary": format_creator_summary_text(res),
            "status": res.get("status") if isinstance(res, dict) else None,
        }
    except Exception as e:
        return {"ok": False, "mode": "creator", "error": str(e)}


def spawn(
    task: str,
    *,
    mode: str = "creator",
    parent_id: str = "",
    tools: Optional[List[BaseTool]] = None,
    project_context: str = "",
    bridge: Any = None,
    max_mini_rounds: int = 12,
) -> str:
    """Start a background sub-agent job; returns token."""
    task = str(task or "").strip()
    token = f"sub_{uuid.uuid4().hex[:10]}"
    ev = threading.Event()
    with _JOBS_LOCK:
        _JOBS[token] = {
            "status": "pending",
            "result": None,
            "error": None,
            "event": ev,
            "task": task,
            "mode": mode,
            "parent_id": parent_id,
            "started": time.time(),
        }

    job_tools = list(tools or [])

    def _worker() -> None:
        _SLOT_SEM.acquire()
        prev_ac = None
        tt = None
        try:
            try:
                from Agent.tools import terminal_tool as tt
                prev_ac = getattr(tt, "AUTO_CONFIRM", False)
                tt.AUTO_CONFIRM = True
            except Exception:
                tt = None  # type: ignore[assignment]

            with _JOBS_LOCK:
                if token in _JOBS:
                    _JOBS[token]["status"] = "running"

            _notify_spawn(token, task, mode, parent_id)

            result = run_subagent_job(
                task,
                mode,
                parent_id,
                job_tools,
                project_context=project_context,
                bridge=bridge,
                max_mini_rounds=max_mini_rounds,
            )

            with _JOBS_LOCK:
                if token not in _JOBS:
                    return
                if result.get("ok"):
                    _JOBS[token]["result"] = result
                    _JOBS[token]["status"] = "done"
                    _notify_done(token, "done", result=result)
                else:
                    err = str(result.get("error") or "subagent_failed")
                    _JOBS[token]["error"] = err
                    _JOBS[token]["status"] = "error"
                    _notify_done(token, "error", error=err)
        except Exception as e:
            with _JOBS_LOCK:
                if token in _JOBS:
                    _JOBS[token]["error"] = str(e)
                    _JOBS[token]["status"] = "error"
            _notify_done(token, "error", error=str(e))
        finally:
            with _JOBS_LOCK:
                if token in _JOBS:
                    _JOBS[token]["event"].set()
            if tt is not None:
                try:
                    tt.AUTO_CONFIRM = bool(prev_ac)
                except Exception:
                    pass
            _SLOT_SEM.release()

    threading.Thread(target=_worker, name=f"subagent-{token}", daemon=True).start()
    return token


def get_result(token: str, wait_seconds: float = 0) -> Dict[str, Any]:
    """Poll or wait for a sub-agent job by token."""
    tok = (token or "").strip()
    with _JOBS_LOCK:
        j = _JOBS.get(tok)
    if not j:
        return {"ok": False, "error": "unknown_subagent_token", "token": tok}

    ev: threading.Event = j.get("event")  # type: ignore[assignment]
    if wait_seconds and wait_seconds > 0 and ev is not None:
        ev.wait(timeout=float(wait_seconds))

    with _JOBS_LOCK:
        j2 = dict(_JOBS.get(tok) or {})

    st = j2.get("status")
    if st in ("pending", "running"):
        return {
            "ok": True,
            "status": "running",
            "token": tok,
            "hint": "Повтори get_subagent_result с wait_seconds > 0.",
        }
    if st == "error":
        return {"ok": False, "error": j2.get("error"), "token": tok}
    return {"ok": True, "status": "done", "data": j2.get("result"), "token": tok}


def running_count() -> int:
    with _JOBS_LOCK:
        return sum(
            1 for j in _JOBS.values()
            if j.get("status") in ("pending", "running")
        )
