"""Generate ``project_brain/`` via Relator templates (or Markdown fallback)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _slug(mid: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", mid, flags=re.UNICODE).strip("_")
    return (s[:120] or "module") + ".md"


def _write_fallback(ctx: Dict[str, Any], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    overview = [
        f"# {ctx.get('project_name', 'project')}",
        "",
        str(ctx.get("project_purpose") or ""),
        "",
        "## Modules (summary)",
        "",
    ]
    for row in (ctx.get("modules") or [])[:80]:
        if isinstance(row, dict):
            overview.append(f"- **{row.get('name')}** ({row.get('type')}): {row.get('purpose', '')[:200]}")
    p = out_dir / "overview.md"
    p.write_text("\n".join(overview), encoding="utf-8")
    written.append(p)
    arch = out_dir / "architecture.md"
    arch.write_text(
        "# Architecture\n\n" + str(ctx.get("architecture", {}).get("overview", "")),
        encoding="utf-8",
    )
    written.append(arch)
    return written


def _relator_render(tpl_dir: Path, template_name: str, out_path: Path,
                    context: Dict[str, Any], save_context: bool = True) -> bool:
    """Render a Relator template to out_path.

    Uses Relator 1.3 ``save_context=True`` to persist ``*.relator-context.json``
    alongside the output for incremental rebuilds.
    """
    try:
        from relator import compile_template
    except ImportError:
        return False
    tpl = tpl_dir / template_name
    if not tpl.is_file():
        return False
    try:
        ctx_path = out_path.with_suffix(".relator-context.json") if save_context else False
        compile_template(tpl, context, out_path, assets_dir=tpl_dir,
                         save_context=ctx_path if save_context else False)
        return True
    except Exception:
        return False


def _load_changelog_entries(root: Path) -> List[Dict[str, Any]]:
    """Load changelog entries from ``.lorne/brain_changelog.jsonl``."""
    changelog_path = root / ".lorne" / "brain_changelog.jsonl"
    entries: List[Dict[str, Any]] = []
    if not changelog_path.is_file():
        return entries
    try:
        for line in changelog_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return entries[-200:]  # Keep last 200 entries


def append_changelog_entry(root: Path, entry: Dict[str, Any]) -> None:
    """Append a single changelog entry to the brain changelog JSONL file."""
    import datetime
    changelog_path = root / ".lorne" / "brain_changelog.jsonl"
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
    try:
        with open(changelog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def build_brain_markdown(root: Path, scan: Dict[str, Any]) -> List[Path]:
    """Render brain tree under ``project_brain/``; return written file paths.

    Uses Relator 1.3 with ``save_context=True`` so each render saves a
    ``*.relator-context.json`` file for incremental rebuilds.
    Loads changelog entries from ``.lorne/brain_changelog.jsonl`` and injects
    them into the context as ``changelog_entries``.
    """
    from .context_builder import build_project_context

    ctx = build_project_context(scan, root)
    # Inject changelog for Relator changelog.md template
    ctx["changelog_entries"] = _load_changelog_entries(root)
    tpl_dir = Path(__file__).resolve().parent / "templates"
    out_dir = root / "project_brain"
    modules_dir = out_dir / "modules"
    machine_dir = out_dir / "machine"
    services_dir = out_dir / "services"
    agents_dir = out_dir / "agents"
    for d in (out_dir, modules_dir, machine_dir, services_dir, agents_dir):
        d.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    def _try(name: str, dest: Path) -> None:
        if _relator_render(tpl_dir, name, dest, ctx, save_context=True):
            written.append(dest)

    _try("overview.md", out_dir / "overview.md")
    _try("architecture.md", out_dir / "architecture.md")
    _try("glossary.md", out_dir / "glossary.md")
    _try("tools.md", out_dir / "tools.md")
    _try("flows.md", out_dir / "flows.md")
    _try("changelog.md", out_dir / "changelog.md")

    if not (out_dir / "overview.md").is_file():
        written.extend(_write_fallback(ctx, out_dir))

    for mod in ctx.get("modules_detail") or []:
        if not isinstance(mod, dict):
            continue
        mid = str(mod.get("name") or "unknown")
        slug = _slug(mid)
        flat: Dict[str, Any] = {
            "module_name": mod.get("name") or "",
            "module_purpose": mod.get("purpose") or "",
            "responsibilities": mod.get("responsibilities") or [],
            "public_api": mod.get("public_api") or [],
            "dependencies": mod.get("dependencies") or [],
            "used_by": mod.get("used_by") or [],
            "side_effects": mod.get("side_effects") or [],
            "risks": mod.get("risks") or [],
            "files": mod.get("files") or [],
            "entrypoints": mod.get("entrypoints") or [],
            "api_hints": mod.get("api_hints") or [],
        }
        dest = modules_dir / slug
        if _relator_render(tpl_dir, "module.md", dest, flat):
            written.append(dest)
        mdest = machine_dir / (slug.replace(".md", ".machine.md"))
        if _relator_render(tpl_dir, "machine.md", mdest, flat):
            written.append(mdest)

    # services / agents stubs (structure for RAG)
    for svc in ctx.get("services") or []:
        if not isinstance(svc, dict):
            continue
        fn = services_dir / f"{_slug(str(svc.get('name', 'service')))}"
        fn.write_text(f"# Service: {svc.get('name')}\n\n{svc.get('description', '')}", encoding="utf-8")
        written.append(fn)
    for ag in ctx.get("agents") or []:
        if not isinstance(ag, dict):
            continue
        fn = agents_dir / f"{_slug(str(ag.get('name', 'agent')))}"
        fn.write_text(f"# Agent: {ag.get('name')}\n\n{ag.get('role', '')}", encoding="utf-8")
        written.append(fn)

    # rag_manifest.json (written in Python — not Relator %% to avoid JSON loop issues)
    brain_files: List[Dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name == "rag_manifest.json":
            continue
        rel = str(p.relative_to(out_dir)).replace("\\", "/")
        try:
            raw = p.read_bytes()
            h = hashlib.sha256(raw).hexdigest()[:16]
        except OSError:
            h = ""
        typ = "md" if p.suffix.lower() == ".md" else p.suffix.lstrip(".") or "file"
        mod_hint = ""
        if "modules/" in rel:
            mod_hint = rel.split("modules/", 1)[-1].replace(".md", "")
        brain_files.append({"path": rel, "type": typ, "module": mod_hint, "hash": h})

    manifest = {
        "project": ctx.get("project_name"),
        "version": ctx.get("version"),
        "generated_at": ctx.get("generated_at"),
        "modules_count": ctx.get("modules_count"),
        "brain_files": brain_files,
    }
    mp = out_dir / "rag_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written.append(mp)

    return sorted(set(written), key=lambda x: str(x))
