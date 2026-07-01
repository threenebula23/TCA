"""Extended mega-tools: one schema per domain to save bind_tools token budget.

Delegates to existing tools (ast_analyze, search_in_files, rag_search, session_notes,
structured_memory, run_command, lint_check, git_diff, list_files, project_brain_tool).
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool

try:
    from .ast_tool import ast_analyze
    from .file_ops import list_files, search_in_files, edit_file
    from .lint_tool import lint_check
    from .memory_tool import structured_memory
    from .notes_tool import session_notes
    from .terminal_tool import run_command
    from .git_tool import git_diff
except ImportError:
    from Agent.tools.ast_tool import ast_analyze
    from Agent.tools.file_ops import list_files, search_in_files, edit_file
    from Agent.tools.lint_tool import lint_check
    from Agent.tools.memory_tool import structured_memory
    from Agent.tools.notes_tool import session_notes
    from Agent.tools.terminal_tool import run_command
    from Agent.tools.git_tool import git_diff


def _rag_invoke(query: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        from Agent.rag import get_rag_tool
    except ImportError:
        from ..rag import get_rag_tool  # type: ignore
    return get_rag_tool().invoke({"query": query, "top_k": top_k})


def _brain_write(
    brain_rel_path: str,
    content: str,
    write_mode: str = "append",
) -> Dict[str, Any]:
    try:
        from .compact_tools import project_brain_tool
    except ImportError:
        from Agent.tools.compact_tools import project_brain_tool
    return project_brain_tool.invoke(
        {
            "action": "write_brain",
            "brain_rel_path": brain_rel_path,
            "content": content,
            "write_mode": write_mode,
        }
    )


def _import_graph(path: str, depth: int = 1) -> Dict[str, Any]:
    """Build a shallow import graph for a Python/JS file."""
    p = Path(path)
    if not p.is_file():
        return {"error": f"File not found: {path}"}
    ast_out = ast_analyze.invoke({"path": path, "query": ""})
    imports = ast_out.get("imports") or []
    graph: List[Dict[str, Any]] = [{"file": str(p), "imports": imports}]
    if depth > 1 and p.suffix.lower() == ".py":
        root = p.parent
        for imp in imports[:15]:
            mod = imp.split()[0].replace("from", "").strip()
            if mod.startswith("."):
                continue
            mod_path = mod.replace(".", "/")
            candidates = list(root.glob(f"{mod_path}.py")) + list(root.glob(f"{mod_path}/__init__.py"))
            for c in candidates[:1]:
                sub = ast_analyze.invoke({"path": str(c), "query": ""})
                graph.append({"file": str(c), "imports": sub.get("imports") or []})
    return {"root": str(p), "depth": depth, "graph": graph}


def _find_symbol(symbol: str, directory: str = ".", file_pattern: str = "*") -> Dict[str, Any]:
    if not symbol.strip():
        return {"error": "symbol required"}
    out = search_in_files.invoke(
        {
            "directory": directory,
            "query": symbol.strip(),
            "file_pattern": file_pattern or "*",
            "max_files": 30,
        }
    )
    out["_action"] = "find_symbol"
    out["symbol"] = symbol.strip()
    return out


def _apply_unified_patch(patch_text: str) -> Dict[str, Any]:
    """Apply a minimal unified diff (---/+++ hunks) to files."""
    if not (patch_text or "").strip():
        return {"ok": False, "error": "patch_text required"}
    lines = patch_text.splitlines()
    files_patched: List[str] = []
    errors: List[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            errors.append(f"malformed header at line {i + 1}")
            break
        old_path = lines[i][4:].split("\t")[0].strip()
        new_path = lines[i + 1][4:].split("\t")[0].strip()
        target = new_path if new_path != "/dev/null" else old_path
        target = target.removeprefix("a/").removeprefix("b/")
        i += 2
        hunks: List[List[str]] = []
        current: List[str] = []
        while i < len(lines) and not lines[i].startswith("--- "):
            if lines[i].startswith("@@"):
                if current:
                    hunks.append(current)
                    current = []
            elif lines[i].startswith((" ", "+", "-")) or lines[i] == "":
                current.append(lines[i])
            i += 1
        if current:
            hunks.append(current)
        try:
            p = Path(target)
            original = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
            result_lines = original.splitlines(keepends=True)
            if not result_lines and original:
                result_lines = [original]
            for hunk in hunks:
                old_chunk = [ln[1:] if ln[:1] in "+-" else ln for ln in hunk if ln[:1] in " -"]
                new_chunk = [ln[1:] if ln[:1] in "+-" else ln for ln in hunk if ln[:1] in " +"]
                old_text = "".join(old_chunk)
                new_text = "".join(new_chunk)
                joined = "".join(result_lines)
                if old_text and old_text in joined:
                    joined = joined.replace(old_text, new_text, 1)
                    result_lines = joined.splitlines(keepends=True)
                    if not result_lines and joined:
                        result_lines = [joined]
                elif not old_text and new_text:
                    result_lines.append(new_text if new_text.endswith("\n") else new_text + "\n")
                else:
                    errors.append(f"hunk not applied in {target}")
            new_content = "".join(result_lines)
            edit_file.invoke({"path": target, "old_str": original, "new_str": new_content})
            files_patched.append(target)
        except Exception as e:
            errors.append(f"{target}: {e}")
    return {"ok": not errors, "patched": files_patched, "errors": errors}


def _transcript_search(query: str, max_hits: int = 10) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    q = (query or "").strip().lower()
    if not q:
        return {"error": "query required", "hits": []}
    roots: List[Path] = []
    try:
        from Agent.path_utils import get_project_root
        proj = get_project_root().resolve()
        slug = str(proj).replace("/", "-").lstrip("-")
        roots.append(Path.home() / ".cursor" / "projects" / slug / "agent-transcripts")
    except Exception:
        pass
    roots.append(Path(".lorne") / "transcripts")
    for root in roots:
        if not root.is_dir():
            continue
        for fp in sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if q not in text.lower():
                continue
            snippet = ""
            for line in text.splitlines():
                if q in line.lower():
                    snippet = line[:240]
                    break
            hits.append({"file": str(fp), "snippet": snippet})
            if len(hits) >= max_hits:
                break
        if hits:
            break
    return {"query": query, "hits": hits, "count": len(hits)}


def _config_inspect(scope: str = "agent") -> Dict[str, Any]:
    out: Dict[str, Any] = {"scope": scope}
    try:
        import Agent.config as cfg_mod
        public = {
            k: getattr(cfg_mod, k)
            for k in dir(cfg_mod)
            if not k.startswith("_") and isinstance(getattr(cfg_mod, k, None), (str, int, float, bool))
        }
        out["config_keys"] = sorted(public.keys())[:40]
    except Exception as e:
        out["config_error"] = str(e)
    try:
        from .env_tool import env_info
    except ImportError:
        from Agent.tools.env_tool import env_info
    env_out = env_info.invoke({})
    out["environment"] = {
        k: env_out.get(k)
        for k in ("python_version", "os", "cwd", "checked_packages", "checked_commands")
        if k in env_out
    }
    return out


def _review_checklist() -> Dict[str, Any]:
    return {
        "checklist": [
            "Run lint/tests on changed paths",
            "Verify imports and public API unchanged unless intended",
            "Check edge cases and error handling",
            "Confirm no secrets or debug prints committed",
            "Update docs if behavior or CLI changed",
            "Smoke-test the happy path manually",
        ],
        "format": "markdown",
    }


def _notify_when_done(message: str, channel: str = "session") -> Dict[str, Any]:
    payload = json.dumps({"message": message, "channel": channel}, ensure_ascii=False)
    return structured_memory.invoke(
        {"action": "set", "key": "notify_when_done", "value": payload, "namespace": "meta"}
    )


def _tools_catalog_list(group: str = "") -> Dict[str, Any]:
    try:
        from Agent.tool_registry import build_tools, _EXTENDED_TOOL_NAMES
    except ImportError:
        from ..tool_registry import build_tools, _EXTENDED_TOOL_NAMES  # type: ignore
    tools, _ = build_tools(extended=False)
    entries: List[Dict[str, str]] = []
    gfilter = (group or "").strip().lower()
    for t in tools:
        name = getattr(t, "name", "") or ""
        desc = (getattr(t, "description", "") or "")[:120]
        tier = "extended" if name in _EXTENDED_TOOL_NAMES else "base"
        if gfilter and gfilter not in tier and gfilter not in name:
            continue
        entries.append({"name": name, "tier": tier, "description": desc})
    return {"count": len(entries), "tools": entries[:80], "group_filter": group or None}


def _tools_catalog_describe(name: str) -> Dict[str, Any]:
    if not (name or "").strip():
        return {"error": "name required for describe"}
    try:
        from Agent.tool_registry import build_tools
    except ImportError:
        from ..tool_registry import build_tools
    tools, _ = build_tools(extended=True)
    for t in tools:
        tname = getattr(t, "name", "") or ""
        if tname == name.strip():
            return {
                "name": tname,
                "description": getattr(t, "description", "") or "",
                "args_schema": str(getattr(t, "args_schema", None)),
            }
    return {"error": f"tool not found: {name}"}


@tool
def code_intel_tool(
    action: str,
    path: str = "",
    query: str = "",
    symbol: str = "",
    directory: str = ".",
    file_pattern: str = "*",
    depth: int = 1,
) -> Dict[str, Any]:
    """Code intel: find_symbol | import_graph | ast.

    find_symbol: symbol + directory; import_graph: path + depth; ast: path + query.
    """
    a = (action or "").strip().lower()
    if a == "find_symbol":
        return _find_symbol(symbol or query, directory, file_pattern)
    if a == "import_graph":
        if not path.strip():
            return {"error": "path required for import_graph"}
        return _import_graph(path, depth=max(1, min(depth, 3)))
    if a == "ast":
        if not path.strip():
            return {"error": "path required for ast"}
        return ast_analyze.invoke({"path": path, "query": query})
    return {"error": "bad_action", "hint": "find_symbol|import_graph|ast"}


@tool
def workspace_search(
    action: str,
    query: str = "",
    directory: str = ".",
    file_pattern: str = "*.py",
    top_k: int = 5,
    tag: str = "",
) -> Dict[str, Any]:
    """Workspace search: brain (RAG) | code (grep) | notes (session_notes)."""
    a = (action or "").strip().lower()
    if not (query or "").strip() and a != "notes":
        return {"error": "query required"}
    if a == "brain":
        return _rag_invoke(query.strip(), top_k=max(1, min(top_k, 20)))
    if a == "code":
        return search_in_files.invoke(
            {
                "directory": directory,
                "query": query.strip(),
                "file_pattern": file_pattern,
                "max_files": 50,
            }
        )
    if a == "notes":
        return session_notes.invoke({"action": "search", "content": query, "tag": tag})
    return {"error": "bad_action", "hint": "brain|code|notes"}


@tool
def net_tool(
    action: str,
    url: str = "",
    host: str = "",
    port: int = 0,
    timeout_seconds: int = 10,
    db_path: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """Network/data: http (fetch URL) | port_check (host + port) | db_query (read-only SQLite SELECT)."""
    a = (action or "").strip().lower()
    if a == "http":
        if not url.strip():
            return {"error": "url required for http"}
        try:
            from .web_tool import web_fetch
        except ImportError:
            from Agent.tools.web_tool import web_fetch
        return web_fetch.invoke({"url": url.strip()})
    if a == "port_check":
        h = (host or url or "").strip()
        if not h or port <= 0:
            return {"error": "host and port required for port_check"}
        try:
            with socket.create_connection((h, int(port)), timeout=min(timeout_seconds, 60)):
                return {"ok": True, "host": h, "port": int(port), "open": True}
        except OSError as e:
            return {"ok": False, "host": h, "port": int(port), "open": False, "error": str(e)}
    if a == "db_query":
        return _db_query(db_path, query)
    return {"error": "bad_action", "hint": "http|port_check|db_query"}


def _db_query(db_path: str, query: str) -> Dict[str, Any]:
    """SQLite-only, read-only helper — no driver deps, no write access.

    Keeping this to SQLite + a strict ``SELECT``/``PRAGMA table_info`` allow-list
    avoids pulling in DB-specific drivers/credentials handling for what the
    plan scoped as a QA/inspection tool, not a general DB client.
    """
    import re
    import sqlite3

    if not db_path.strip():
        return {"error": "db_path required for db_query"}
    q = (query or "").strip()
    if not q:
        return {"error": "query required for db_query"}
    if not re.match(r"^(select|pragma\s+table_info)\b", q, re.IGNORECASE):
        return {"error": "only SELECT / PRAGMA table_info queries are allowed"}
    try:
        from Agent.path_utils import resolve_abs_path

        path = resolve_abs_path(db_path.strip())
    except Exception:
        path = db_path.strip()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(q)
            rows = [dict(r) for r in cur.fetchmany(200)]
            return {"ok": True, "columns": list(rows[0].keys()) if rows else [], "rows": rows, "truncated": len(rows) == 200}
        finally:
            conn.close()
    except Exception as e:
        return {"error": str(e)}


@tool
def viz_tool(
    action: str,
    spec_json: str = "",
    title: str = "",
    mermaid: str = "",
) -> Dict[str, Any]:
    """Viz helper: chart (labels/series JSON) | diagram (mermaid-lite text).

    Returns data pre-shaped for the TUI's fenced-block renderers — put the
    result straight into your final answer as ```lorne-chart\\n<chart json>\\n```
    or ```mermaid\\n<diagram text>\\n``` (see h-viz-tools-prompt: the TUI has no
    Vega-Lite renderer, only a plotext ASCII one, so this must match that
    ``{labels, series}`` shape, not a Vega-Lite spec).
    """
    a = (action or "").strip().lower()
    if a == "chart":
        try:
            spec = json.loads(spec_json) if spec_json.strip() else {}
        except json.JSONDecodeError as e:
            return {"error": f"invalid spec_json: {e}"}
        if not isinstance(spec, dict) or not spec.get("series"):
            return {
                "error": "spec_json must be JSON like "
                '{"labels": ["a","b"], "series": [{"name": "x", "values": [1,2], "type": "bar"}]}'
            }
        if title:
            spec.setdefault("title", title)
        return {"format": "lorne-chart", "chart": spec}
    if a == "diagram":
        body = (mermaid or spec_json or "").strip()
        if not body:
            return {"error": "mermaid text required for diagram"}
        if not body.startswith("graph") and not body.startswith("flowchart"):
            body = f"flowchart TD\n{body}"
        return {"format": "mermaid", "diagram": body}
    return {"error": "bad_action", "hint": "chart|diagram"}


@tool
def qa_extended_tool(
    action: str,
    path: str = "",
    command: str = "",
    fix: bool = False,
    linter: str = "auto",
    cwd: str = "",
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """QA: test (pytest/run_command) | lint (lint_check)."""
    a = (action or "").strip().lower()
    if a == "lint":
        if not path.strip():
            return {"error": "path required for lint"}
        return lint_check.invoke({"path": path, "fix": fix, "linter": linter})
    if a == "test":
        cmd = (command or "").strip() or "python -m pytest -q"
        if path.strip() and "pytest" in cmd and path not in cmd:
            cmd = f"{cmd} {path.strip()}"
        return run_command.invoke(
            {"command": cmd, "cwd": cwd, "timeout_seconds": timeout_seconds, "background": False}
        )
    return {"error": "bad_action", "hint": "test|lint"}


@tool
def session_meta_tool(
    action: str,
    query: str = "",
    json_text: str = "",
    message: str = "",
    scope: str = "agent",
    channel: str = "session",
) -> Dict[str, Any]:
    """Session meta: transcript_search | config_inspect | json_validate | review_checklist | notify_when_done."""
    a = (action or "").strip().lower()
    if a == "transcript_search":
        return _transcript_search(query)
    if a == "config_inspect":
        return _config_inspect(scope)
    if a == "json_validate":
        if not json_text.strip():
            return {"error": "json_text required"}
        try:
            parsed = json.loads(json_text)
            return {"ok": True, "type": type(parsed).__name__, "preview": str(parsed)[:500]}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": str(e)}
    if a == "review_checklist":
        return _review_checklist()
    if a == "notify_when_done":
        if not message.strip():
            return {"error": "message required for notify_when_done"}
        return _notify_when_done(message.strip(), channel)
    return {
        "error": "bad_action",
        "hint": "transcript_search|config_inspect|json_validate|review_checklist|notify_when_done",
    }


@tool
def tools_catalog(action: str, name: str = "", group: str = "") -> Dict[str, Any]:
    """Tools catalog: list (optional group filter) | describe (name)."""
    a = (action or "").strip().lower()
    if a == "list":
        return _tools_catalog_list(group)
    if a == "describe":
        return _tools_catalog_describe(name)
    return {"error": "bad_action", "hint": "list|describe"}


@tool
def diff_tool(commit: str = "", path: str = "") -> Dict[str, Any]:
    """Git diff: commit hash or empty for working tree changes."""
    out = git_diff.invoke({"commit": commit})
    if path.strip() and isinstance(out.get("diff"), str):
        lines = [ln for ln in out["diff"].splitlines() if path in ln or ln.startswith(("+", "-", " "))]
        out["diff"] = "\n".join(lines[:500])
    return out


@tool
def apply_patch(patch_text: str) -> Dict[str, Any]:
    """Apply unified diff patch text to workspace files."""
    return _apply_unified_patch(patch_text)


@tool
def project_tree(path: str = ".", pattern: str = "*", max_entries: int = 500) -> Dict[str, Any]:
    """Recursive project tree via list_files."""
    out = list_files.invoke({"path": path, "recursive": True, "pattern": pattern})
    entries = out.get("entries") or []
    if len(entries) > max_entries:
        out["entries"] = entries[:max_entries]
        out["truncated"] = True
    return out


@tool
def brain_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search Project Brain / RAG index."""
    return _rag_invoke(query, top_k=max(1, min(top_k, 20)))


@tool
def export_to_brain(
    brain_rel_path: str,
    content: str,
    write_mode: str = "append",
) -> Dict[str, Any]:
    """Write markdown into project_brain (delegates project_brain_tool)."""
    return _brain_write(brain_rel_path, content, write_mode)


@tool
def memory_search(query: str = "", namespace: str = "default") -> Dict[str, Any]:
    """Search structured_memory entries by substring in keys/values."""
    listed = structured_memory.invoke({"action": "list", "namespace": namespace})
    entries = listed.get("entries") or {}
    q = (query or "").strip().lower()
    if not q:
        return listed
    matched = {
        k: v for k, v in entries.items()
        if q in str(k).lower() or q in str(v).lower()
    }
    return {"namespace": namespace, "query": query, "entries": matched, "count": len(matched)}
