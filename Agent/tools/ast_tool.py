"""AST-based code structure analysis tool."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


def _analyze_python(source: str, query: str = "") -> Dict[str, Any]:
    """Parse Python source with stdlib ast; return classes, functions, imports."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}", "language": "python"}

    classes: List[Dict] = []
    functions: List[Dict] = []
    imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                {
                    "name": m.name,
                    "args": [a.arg for a in m.args.args],
                    "line": m.lineno,
                }
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not any(
                isinstance(p, ast.ClassDef) and any(
                    isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == node.name
                    for c in p.body
                )
                for p in ast.walk(tree)
                if isinstance(p, ast.ClassDef)
            ):
                functions.append({
                    "name": node.name,
                    "args": [a.arg for a in node.args.args],
                    "line": node.lineno,
                    "docstring": (ast.get_docstring(node) or "")[:120],
                })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            imports.append(f"from {mod} import {', '.join(names)}")

    result = {
        "language": "python",
        "total_lines": len(source.splitlines()),
        "classes": classes,
        "functions": functions,
        "imports": imports[:30],
    }

    if query:
        q = query.lower()
        result["classes"] = [c for c in classes if q in c["name"].lower()]
        result["functions"] = [f for f in functions if q in f["name"].lower()]

    return result


def _analyze_js_ts(source: str, query: str = "") -> Dict[str, Any]:
    """Regex-based JS/TS heuristic analysis (no external deps)."""
    classes = re.findall(r"class\s+(\w+)", source)
    functions = re.findall(r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()", source)
    imports = re.findall(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", source)
    exports = re.findall(r"export\s+(?:default\s+)?(?:class|function|const|let)\s+(\w+)", source)

    funcs = [f[0] or f[1] for f in functions if f[0] or f[1]]

    if query:
        q = query.lower()
        classes = [c for c in classes if q in c.lower()]
        funcs = [f for f in funcs if q in f.lower()]

    return {
        "language": "javascript/typescript",
        "total_lines": len(source.splitlines()),
        "classes": [{"name": c} for c in classes],
        "functions": [{"name": f} for f in funcs],
        "imports": imports[:20],
        "exports": exports[:20],
    }


def _analyze_markdown(source: str) -> Dict[str, Any]:
    """Extract headings and code blocks from Markdown."""
    headings = []
    for m in re.finditer(r"^(#{1,6})\s+(.+)", source, re.MULTILINE):
        headings.append({"level": len(m.group(1)), "title": m.group(2).strip()})
    code_langs = re.findall(r"```(\w*)", source)
    return {
        "language": "markdown",
        "total_lines": len(source.splitlines()),
        "headings": headings,
        "code_block_languages": code_langs,
    }


@tool
def ast_analyze(path: str, query: str = "") -> dict:
    """AST-анализ Python/JS/TS/MD файла: классы, функции, импорты, сигнатуры.

    path: путь к файлу
    query: опционально фильтрует вывод (например 'class Auth' или 'def handle').
    Возвращает структуру без чтения всего файла целиком — экономит токены.
    Поддерживает Python (stdlib ast), JS/TS (regex), Markdown (заголовки).
    """
    p = Path(path)
    if not p.is_file():
        return {"error": f"File not found: {path}"}
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": str(e)}

    ext = p.suffix.lower()
    if ext == ".py":
        return _analyze_python(source, query)
    elif ext in (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"):
        return _analyze_js_ts(source, query)
    elif ext in (".md", ".mdx"):
        return _analyze_markdown(source)
    else:
        lines = source.splitlines()
        return {
            "language": "unknown",
            "total_lines": len(lines),
            "preview": "\n".join(lines[:10]),
        }
