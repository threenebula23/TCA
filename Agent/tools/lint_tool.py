"""Run linters and return structured error reports."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool


def _detect_linter(path: str, linter: str) -> str:
    if linter != "auto":
        return linter
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return "ruff"
    if ext in (".js", ".ts", ".tsx", ".jsx"):
        return "eslint"
    return "ruff"


def _run_ruff(path: str, fix: bool) -> Dict[str, Any]:
    cmd = ["python", "-m", "ruff", "check", path, "--output-format=json"]
    if fix:
        cmd.append("--fix")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        errors: List[Dict] = []
        try:
            raw = json.loads(result.stdout or "[]")
            for item in raw:
                errors.append({
                    "file": item.get("filename", path),
                    "line": item.get("location", {}).get("row", 0),
                    "col": item.get("location", {}).get("column", 0),
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                })
        except Exception:
            pass
        return {
            "ok": result.returncode == 0,
            "errors": errors,
            "tool_used": "ruff",
            "fixed": fix,
        }
    except FileNotFoundError:
        return {"error": "ruff not found; install with: pip install ruff", "tool_used": "ruff"}
    except Exception as e:
        return {"error": str(e), "tool_used": "ruff"}


def _run_pylint(path: str) -> Dict[str, Any]:
    cmd = ["python", "-m", "pylint", path, "--output-format=json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        errors: List[Dict] = []
        try:
            raw = json.loads(result.stdout or "[]")
            for item in raw:
                errors.append({
                    "file": item.get("path", path),
                    "line": item.get("line", 0),
                    "col": item.get("column", 0),
                    "code": item.get("message-id", ""),
                    "message": item.get("message", ""),
                })
        except Exception:
            pass
        return {"ok": result.returncode == 0, "errors": errors, "tool_used": "pylint"}
    except FileNotFoundError:
        return {"error": "pylint not found; install with: pip install pylint", "tool_used": "pylint"}
    except Exception as e:
        return {"error": str(e), "tool_used": "pylint"}


@tool
def lint_check(path: str, fix: bool = False, linter: str = "auto") -> dict:
    """Запускает линтер на path и возвращает ошибки структурировано.

    path: путь к файлу или директории
    fix: если True — применить автоисправления (ruff --fix)
    linter: auto | ruff | pylint | eslint | tsc
    Всегда вызывай после правки кода перед тем как доложить что готово.
    Возвращает {ok, errors: [{file, line, col, code, message}], tool_used}.
    """
    used_linter = _detect_linter(path, linter)

    if used_linter == "ruff":
        return _run_ruff(path, fix)
    elif used_linter == "pylint":
        return _run_pylint(path)
    elif used_linter in ("eslint", "tsc"):
        cmd = [used_linter, path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
                "tool_used": used_linter,
            }
        except FileNotFoundError:
            return {"error": f"{used_linter} not found", "tool_used": used_linter}
    else:
        return {"error": f"Unknown linter: {used_linter}"}
