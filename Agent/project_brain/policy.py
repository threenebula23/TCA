"""Единая политика Project Brain по режимам чата (bootstrap/read/write/refresh/reindex).

Раньше эти решения были размазаны по ``graph_runner.py``, ``creator_mode.py``,
``deep_solver/legacy_loop.py`` и TUI-коду, каждый со своими условиями и
``except Exception: pass``. Здесь — одна таблица + debounce-состояние для
полного скана (дорогая операция), чтобы её не гонять на каждый ход.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ``export_session_notes``: тег session_notes, который стоит переносить в
# ``agent/<tag>_notes.md`` (P1-3 / mode-*-brain-export). ``False`` — не переносить.
BRAIN_POLICY: Dict[str, Dict[str, Any]] = {
    "agent": {
        "bootstrap": True,
        "reindex_per_round": False,
        "refresh_on_end": False,
        "require_write_brain": False,
        "export_session_notes": False,
        "force_tools": False,
    },
    "ask": {
        "bootstrap": True,
        "reindex_per_round": False,
        "refresh_on_end": False,
        "write": False,
        "require_write_brain": False,
        "export_session_notes": False,
        "force_tools": ("rag_search",),
    },
    "research": {
        "bootstrap": True,
        "reindex_per_round": True,
        "refresh_on_end": False,
        "require_write_brain": False,
        "export_session_notes": "research",
        "force_tools": False,
    },
    "brainer": {
        "bootstrap": True,
        "reindex_per_round": True,
        "refresh_on_end": True,
        "require_write_brain": True,
        "export_session_notes": False,
        "force_tools": ("rag_search", "project_brain_tool"),
    },
    "creator": {
        "bootstrap": True,
        "reindex_per_round": True,
        "refresh_on_end": "if_code_changed",
        "require_write_brain": False,
        "documenter_write": True,
        "export_session_notes": False,
        "force_tools": False,
    },
    "deep": {
        "bootstrap": True,
        "reindex_per_round": False,
        "refresh_on_end": False,
        "require_write_brain": False,
        "export_final_report": True,
        "export_session_notes": False,
        "force_tools": False,
    },
}

_DEFAULT_MODE = "agent"


def get_policy(mode: Optional[str]) -> Dict[str, Any]:
    """Return the policy dict for ``mode`` (falls back to ``agent``)."""
    m = (mode or _DEFAULT_MODE).strip().lower()
    return BRAIN_POLICY.get(m, BRAIN_POLICY[_DEFAULT_MODE])


def should_reindex_per_round(mode: Optional[str]) -> bool:
    return bool(get_policy(mode).get("reindex_per_round"))


def should_refresh_on_end(mode: Optional[str], *, code_changed: bool = False) -> bool:
    val = get_policy(mode).get("refresh_on_end")
    if val == "if_code_changed":
        return bool(code_changed)
    return bool(val)


def forced_tool_names(mode: Optional[str]) -> tuple:
    val = get_policy(mode).get("force_tools")
    return tuple(val) if val else ()


# ─── Debounce state for full refresh (B1) ──────────────────────────

_STATE_REL = Path(".lorne") / "brain_sync_state.json"
_MIN_FULL_REFRESH_INTERVAL_SEC = 120  # env override: LORNE_BRAIN_REFRESH_DEBOUNCE_SEC


def _state_path(root: Path) -> Path:
    return Path(root) / _STATE_REL


def _load_state(root: Path) -> Dict[str, Any]:
    p = _state_path(root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace")) or {}
    except Exception:
        return {}


def _save_state(root: Path, state: Dict[str, Any]) -> None:
    p = _state_path(root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _changelog_line_count(root: Path) -> int:
    cl = Path(root) / ".lorne" / "brain_changelog.jsonl"
    if not cl.is_file():
        return 0
    try:
        return sum(1 for _ in cl.open("r", encoding="utf-8", errors="replace"))
    except Exception:
        return 0


def _debounce_interval() -> float:
    try:
        from Agent.runtime_paths import env_pref

        raw = env_pref("BRAIN_REFRESH_DEBOUNCE_SEC", "")
        if raw.strip():
            return float(raw.strip())
    except Exception:
        pass
    return float(_MIN_FULL_REFRESH_INTERVAL_SEC)


def should_full_refresh(root: Path, *, force: bool = False) -> bool:
    """True if a full ``refresh_project_brain`` scan is warranted right now.

    Skips redundant scans when nothing changed and the last full refresh was
    recent — the scan (AST across the whole repo + optional Relator render)
    is the most expensive brain operation, and Brainer used to run it after
    *every* completed turn regardless of whether any file changed.
    """
    if force:
        return True
    root = Path(root)
    state = _load_state(root)
    last_ts = float(state.get("last_full_refresh_ts") or 0.0)
    last_changelog_n = int(state.get("last_changelog_lines") or 0)
    now = time.time()
    changelog_n = _changelog_line_count(root)
    if changelog_n != last_changelog_n:
        return True
    if not (root / "project_brain" / "overview.md").is_file():
        return True
    if now - last_ts >= _debounce_interval():
        return True
    return False


def mark_full_refresh_done(root: Path) -> None:
    root = Path(root)
    state = _load_state(root)
    state["last_full_refresh_ts"] = time.time()
    state["last_changelog_lines"] = _changelog_line_count(root)
    _save_state(root, state)
