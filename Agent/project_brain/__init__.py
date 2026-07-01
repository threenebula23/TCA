"""Project Brain: static scan + Markdown output for RAG (optional Relator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .scanner import scan_project
from .build import build_brain_markdown
from .context_builder import build_project_context
from .agent_architecture import (
    AGENT_ARCHITECTURE_FILE,
    reindex_brain_rag,
    write_agent_architecture,
    write_brain_markdown,
)
from .policy import (
    BRAIN_POLICY,
    get_policy,
    mark_full_refresh_done,
    should_full_refresh,
    should_refresh_on_end,
    should_reindex_per_round,
)


def _resolve_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    try:
        from Agent.path_utils import get_project_root

        return get_project_root().resolve()
    except Exception:
        return Path.cwd().resolve()


def bootstrap_project_brain(root: Path | None = None) -> Dict[str, Any]:
    """Create ``project_brain/`` if it doesn't exist yet (chicken-and-egg fix).

    Previously ``project_brain/`` was only ever created by an explicit
    ``project_brain_tool`` call or by switching into Brainer mode — a user
    who stays in the default Agent mode would never get a populated brain.
    Called once at TUI/classic startup; cheap no-op if the brain already
    exists (checks for ``overview.md`` only, not a full scan).
    """
    r = _resolve_root(root)
    marker = r / "project_brain" / "overview.md"
    if marker.is_file():
        return {"bootstrapped": False}
    try:
        summary = refresh_project_brain(r)
        mark_full_refresh_done(r)
        return {"bootstrapped": True, **summary}
    except Exception as e:
        return {"bootstrapped": False, "error": f"{type(e).__name__}: {e}"}


def refresh_project_brain(root: Path | None = None) -> Dict[str, Any]:
    """Scan ``root``, write ``project_brain/**``, return summary paths."""
    if root is None:
        try:
            from Agent.path_utils import get_project_root

            r = get_project_root().resolve()
        except Exception:
            r = Path.cwd().resolve()
    else:
        r = Path(root).resolve()
    data = scan_project(r)
    paths = build_brain_markdown(r, data)
    return {"root": str(r), "written": [str(p) for p in paths], "context": build_project_context(data, r)}


def read_brain_context(
    root: Path | None = None,
    max_chars: int = 1800,
    *,
    include_modules: bool = False,
) -> str:
    """Read a short, prioritised excerpt of ``project_brain/*.md`` for prompt injection.

    Single implementation replacing three near-identical copies that used to
    live in ``project_brain/__init__.py`` (``read_brain_context_summary``),
    ``creator_mode.py`` (``_read_brain_context_for_creator``, budget 4000) and
    ``deep_solver/legacy_loop.py`` (``_read_brain_context``, budget 6000) —
    same priority-file logic, just different ``max_chars``. Callers pass their
    own budget; ``include_modules`` additionally folds in a few
    ``modules/*.md`` files for callers that want more structural detail
    (Creator/Deep previously did this ad hoc).
    """
    try:
        r = _resolve_root(root)
        brain_dir = r / "project_brain"
        if not brain_dir.is_dir():
            return ""
        priority = ["overview.md", "architecture.md", "agent_architecture.md", "modules.md"]
        files = []
        for name in priority:
            p = brain_dir / name
            if p.is_file():
                files.append(p)
        for p in sorted(brain_dir.glob("*.md")):
            if p not in files:
                files.append(p)
        if include_modules:
            mod_dir = brain_dir / "modules"
            if mod_dir.is_dir():
                for p in sorted(mod_dir.glob("*.md"))[:20]:
                    files.append(p)
        parts = []
        total = 0
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    continue
                chunk = f"## {fp.name}\n{text}"
                if total + len(chunk) > max_chars:
                    remaining = max_chars - total
                    if remaining > 200:
                        parts.append(chunk[:remaining] + "\n… (обрезано)")
                    break
                parts.append(chunk)
                total += len(chunk)
            except Exception:
                continue
        return "\n\n".join(parts)
    except Exception:
        return ""


def read_brain_context_summary(root: Path | None = None, max_chars: int = 1800) -> str:
    """Back-compat alias for :func:`read_brain_context`."""
    return read_brain_context(root, max_chars)
