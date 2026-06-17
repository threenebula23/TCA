"""Apply multiple find-and-replace operations to a file in one call."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def batch_replace(path: str, replacements: List[Dict[str, str]], regex: bool = False) -> dict:
    """Применяет несколько замен к файлу за один вызов.

    path: путь к файлу
    replacements: [{"from": "old_name", "to": "new_name"}, ...]
    regex: если True — 'from' интерпретируется как regex
    Идеально для рефакторинга: переименование символа, замена импортов и т.д.
    Возвращает {ok, total_replacements, per_rule: [{from, to, count}], backup_path}.
    """
    p = Path(path)
    if not p.is_file():
        return {"error": f"File not found: {path}"}
    if not replacements:
        return {"error": "replacements list is empty"}

    try:
        original = p.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"Cannot read file: {e}"}

    backup_path = str(p) + ".bak"
    try:
        shutil.copy2(str(p), backup_path)
    except Exception:
        backup_path = ""

    text = original
    per_rule: List[Dict[str, Any]] = []
    total = 0

    for rule in replacements[:50]:
        find = str(rule.get("from", ""))
        replace = str(rule.get("to", ""))
        if not find:
            continue
        try:
            if regex:
                new_text, count = re.subn(find, replace, text)
            else:
                count = text.count(find)
                new_text = text.replace(find, replace)
            text = new_text
            total += count
            per_rule.append({"from": find, "to": replace, "count": count})
        except Exception as e:
            per_rule.append({"from": find, "to": replace, "error": str(e)})

    try:
        p.write_text(text, encoding="utf-8")
    except Exception as e:
        return {"error": f"Cannot write file: {e}", "backup_path": backup_path}

    return {
        "ok": True,
        "total_replacements": total,
        "per_rule": per_rule,
        "backup_path": backup_path,
    }
