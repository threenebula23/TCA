"""Spawn and collect background sub-agents (Creator or mini LangGraph loop)."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

try:
    from ..subagent_runner import spawn, get_result
except ImportError:
    from Agent.subagent_runner import spawn, get_result


def _build_session_tools() -> list:
    try:
        from Agent.tool_registry import build_tools, set_tool_session_prefs
    except ImportError:
        from tool_registry import build_tools, set_tool_session_prefs

    try:
        from Interface.ui_prefs import load_prefs
        prefs = load_prefs()
        am = False
        pw = bool(prefs.get("playwright_python_enabled", False))
        bw = bool(prefs.get("browser_tools_enabled", True))
    except Exception:
        am, pw, bw = False, False, True

    set_tool_session_prefs(agent_mode=am, ask_mode=False, playwright_python=pw, browser_tools=bw)
    tools, _ = build_tools(agent_mode=am, ask_mode=False, playwright_python=pw, browser_tools=bw)
    return tools


@tool
def spawn_subagent(
    task: str,
    mode: str = "creator",
    parent_id: str = "",
) -> Dict[str, Any]:
    """Запустить субагента в фоне (Creator Mode или mini LangGraph). Вернёт token.

    mode: creator (по умолчанию) или mini (быстрый микро-цикл LLM+тулы).
    parent_id: необязательный id родительского агента для UI.
    """
    task = str(task or "").strip()
    if not task:
        return {"ok": False, "error": "empty_task"}

    tools = _build_session_tools()
    token = spawn(
        task,
        mode=mode,
        parent_id=parent_id,
        tools=tools,
    )
    return {
        "ok": True,
        "async": True,
        "token": token,
        "mode": (mode or "creator").strip().lower(),
        "hint": (
            f"Sub-agent в фоне. Вызови get_subagent_result(token='{token}', wait_seconds=...). "
            "Пока можешь вызвать долгий run_command."
        ),
    }


@tool
def get_subagent_result(token: str, wait_seconds: int = 0) -> Dict[str, Any]:
    """Статус или ожидание субагента по token из spawn_subagent. wait_seconds=0 — без ожидания."""
    tok = (token or "").strip()
    if not tok:
        return {"ok": False, "error": "token_required"}
    return get_result(tok, float(wait_seconds or 0))
