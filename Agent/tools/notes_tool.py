"""Free-form session notes and hypotheses scratch pad."""
from __future__ import annotations

import datetime
from pathlib import Path

from langchain_core.tools import tool


def _notes_path() -> Path:
    return Path(".lorne") / "session_notes.md"


@tool
def session_notes(action: str, content: str = "", tag: str = "") -> dict:
    """Свободные заметки сессии: append/read/clear/search.

    action: append | read | clear | search
    content: текст заметки (для append)
    tag: категория ('bug', 'decision', 'todo', 'observation', 'hypothesis', 'fact')
    Записывай сюда рассуждения, гипотезы, промежуточные наблюдения.
    Хранится в .lorne/session_notes.md как Markdown с временными метками.
    """
    p = _notes_path()
    act = (action or "").strip().lower()

    if act == "append":
        if not content:
            return {"error": "content required for append"}
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        tag_str = f" `[{tag}]`" if tag else ""
        entry = f"\n## {ts}{tag_str}\n\n{content.strip()}\n"
        with open(p, "a", encoding="utf-8") as f:
            f.write(entry)
        return {"ok": True, "timestamp": ts, "tag": tag}

    elif act == "read":
        if not p.is_file():
            return {"content": "", "note": "No session notes yet"}
        text = p.read_text(encoding="utf-8")
        return {"content": text, "length": len(text)}

    elif act == "clear":
        if p.is_file():
            p.unlink()
        return {"ok": True, "cleared": True}

    elif act == "search":
        if not p.is_file():
            return {"entries": [], "count": 0}
        text = p.read_text(encoding="utf-8")
        query = (content or tag or "").lower()
        if not query:
            return {"content": text, "note": "No search query; returning all notes"}
        entries = text.split("\n## ")[1:]
        matched = [e for e in entries if query in e.lower()]
        return {
            "entries": [f"## {e}" for e in matched[:20]],
            "count": len(matched),
        }

    return {"error": f"Unknown action '{action}'. Valid: append|read|clear|search"}
