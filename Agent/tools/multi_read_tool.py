"""Read multiple files in a single tool call."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from langchain_core.tools import tool


@tool
def multi_read(paths: List[str], max_lines_each: int = 200) -> dict:
    """Читает несколько файлов одним вызовом (до 8 штук, max_lines_each строк каждый).

    paths: список путей к файлам (максимум 8)
    max_lines_each: максимум строк на файл (по умолчанию 200)
    Возвращает {path: {content, total_lines, truncated}} для каждого файла.
    Снижает количество круговых обращений при анализе взаимосвязанных файлов.
    """
    if not paths:
        return {"error": "paths list is empty"}

    limited_paths = paths[:8]
    results: Dict[str, Dict] = {}

    for raw_path in limited_paths:
        p = Path(raw_path)
        if not p.is_file():
            results[raw_path] = {"error": f"File not found: {raw_path}"}
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            truncated = total > max_lines_each
            content = "\n".join(lines[:max_lines_each])
            results[raw_path] = {
                "content": content,
                "total_lines": total,
                "truncated": truncated,
                "shown_lines": min(total, max_lines_each),
            }
        except Exception as e:
            results[raw_path] = {"error": str(e)}

    return {"files": results, "count": len(results)}
