"""Post-edit result verification tool."""
from __future__ import annotations

import importlib.util
import py_compile
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool


def _check_syntax(path: str) -> Dict[str, Any]:
    try:
        py_compile.compile(path, doraise=True)
        return {"type": "syntax", "path": path, "ok": True}
    except py_compile.PyCompileError as e:
        return {"type": "syntax", "path": path, "ok": False, "error": str(e)}
    except Exception as e:
        return {"type": "syntax", "path": path, "ok": False, "error": str(e)}


def _check_contains(path: str, text: str, negate: bool = False) -> Dict[str, Any]:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        found = text in content
        ok = not found if negate else found
        return {
            "type": "not_contains" if negate else "contains",
            "path": path,
            "text": text[:80],
            "ok": ok,
        }
    except Exception as e:
        return {"type": "contains", "path": path, "ok": False, "error": str(e)}


def _check_import(module: str) -> Dict[str, Any]:
    spec = importlib.util.find_spec(module.replace("-", "_"))
    return {"type": "import", "module": module, "ok": spec is not None}


def _check_file_exists(path: str) -> Dict[str, Any]:
    return {"type": "file_exists", "path": path, "ok": Path(path).exists()}


def _check_command(cmd: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return {"type": "command", "cmd": cmd[:80], "ok": result.returncode == 0, "stdout": result.stdout[:200]}
    except subprocess.TimeoutExpired:
        return {"type": "command", "cmd": cmd[:80], "ok": False, "error": "timeout"}
    except Exception as e:
        return {"type": "command", "cmd": cmd[:80], "ok": False, "error": str(e)}


@tool
def verify_result(checks: List[Dict]) -> dict:
    """Проверка результата после правок: синтаксис, наличие строк, импорты, файлы.

    checks: список проверок, каждая из которых — dict с ключом 'type':
      {"type": "syntax",       "path": "foo.py"}
      {"type": "contains",     "path": "foo.py",  "text": "def handle"}
      {"type": "not_contains", "path": "foo.py",  "text": "old_name"}
      {"type": "import",       "module": "fastapi"}
      {"type": "file_exists",  "path": "bar.py"}
      {"type": "command",      "cmd":  "python -c 'import app'"}

    Всегда вызывай в конце выполнения задачи перед тем как доложить о готовности.
    Возвращает {all_passed, results: [{type, ok, ...}]}.
    """
    results: List[Dict[str, Any]] = []

    for check in checks[:20]:
        ctype = str(check.get("type", "")).lower()
        if ctype == "syntax":
            results.append(_check_syntax(str(check.get("path", ""))))
        elif ctype == "contains":
            results.append(_check_contains(str(check.get("path", "")), str(check.get("text", ""))))
        elif ctype == "not_contains":
            results.append(_check_contains(str(check.get("path", "")), str(check.get("text", "")), negate=True))
        elif ctype == "import":
            results.append(_check_import(str(check.get("module", ""))))
        elif ctype == "file_exists":
            results.append(_check_file_exists(str(check.get("path", ""))))
        elif ctype == "command":
            results.append(_check_command(str(check.get("cmd", ""))))
        else:
            results.append({"type": ctype, "ok": False, "error": f"Unknown check type: {ctype}"})

    all_passed = all(r.get("ok", False) for r in results)
    return {
        "all_passed": all_passed,
        "passed": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }
