"""Heuristic task decomposition tool for weak models."""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool

_PATTERNS: Dict[str, List[Dict[str, str]]] = {
    "code": [
        {"step": "1. Проанализируй задачу", "type": "analyze", "suggested_tool": "ast_analyze или read_file", "description": "Прочитай и пойми текущий код"},
        {"step": "2. Составь план", "type": "plan", "suggested_tool": "save_plan", "description": "Запиши конкретные шаги реализации"},
        {"step": "3. Реализуй изменения", "type": "edit", "suggested_tool": "edit_file или replace_file_lines", "description": "Внеси изменения файл за файлом"},
        {"step": "4. Проверь синтаксис", "type": "verify", "suggested_tool": "lint_check", "description": "Проверь код на ошибки линтером"},
        {"step": "5. Запусти тесты", "type": "test", "suggested_tool": "run_package_script или run_command pytest", "description": "Убедись что тесты проходят"},
        {"step": "6. Финальная проверка", "type": "verify", "suggested_tool": "verify_result", "description": "Проверь ключевые условия задачи"},
    ],
    "research": [
        {"step": "1. Поиск в интернете", "type": "research", "suggested_tool": "web_search", "description": "Найди релевантные материалы"},
        {"step": "2. Чтение страниц", "type": "read", "suggested_tool": "web_fetch", "description": "Прочитай выбранные источники"},
        {"step": "3. Анализ кода проекта", "type": "analyze", "suggested_tool": "multi_read или ast_analyze", "description": "Изучи код проекта"},
        {"step": "4. Фиксация находок", "type": "note", "suggested_tool": "session_notes", "description": "Запиши ключевые выводы"},
        {"step": "5. Составь отчёт", "type": "report", "suggested_tool": "write_file", "description": "Оформи выводы в документ"},
    ],
    "docs": [
        {"step": "1. Прочитай код", "type": "read", "suggested_tool": "multi_read или ast_analyze", "description": "Изучи публичный API"},
        {"step": "2. Прочитай существующую документацию", "type": "read", "suggested_tool": "read_file", "description": "Пойми текущий контекст"},
        {"step": "3. Напиши документацию", "type": "edit", "suggested_tool": "write_file или edit_file", "description": "Создай/обнови документы"},
        {"step": "4. Добавь примеры", "type": "edit", "suggested_tool": "edit_file", "description": "Добавь примеры использования"},
    ],
    "refactor": [
        {"step": "1. Анализ текущего кода", "type": "analyze", "suggested_tool": "ast_analyze + multi_read", "description": "Пойми структуру перед рефакторингом"},
        {"step": "2. Найди все вхождения", "type": "analyze", "suggested_tool": "search_in_files", "description": "Найди где используется изменяемый код"},
        {"step": "3. Применяй изменения файл за файлом", "type": "edit", "suggested_tool": "batch_replace или replace_file_lines", "description": "Рефакторинг по одному файлу"},
        {"step": "4. Проверь линтером", "type": "verify", "suggested_tool": "lint_check", "description": "Убедись что нет синтаксических ошибок"},
        {"step": "5. Запусти тесты", "type": "test", "suggested_tool": "run_command pytest", "description": "Проверь что рефакторинг не сломал поведение"},
    ],
}


def _detect_mode(task: str) -> str:
    t = task.lower()
    if any(w in t for w in ("документа", "readme", "wiki", "docs", "описа")):
        return "docs"
    if any(w in t for w in ("рефактор", "перенос", "переименов", "renam", "refactor", "reorganiz")):
        return "refactor"
    if any(w in t for w in ("исследуй", "найди", "изучи", "research", "find out", "what is")):
        return "research"
    return "code"


@tool
def task_decompose(task: str, context_files: Optional[List[str]] = None, mode: str = "auto") -> dict:
    """Декомпозиция задачи на атомарные шаги (без LLM, на эвристиках).

    task: описание задачи
    context_files: список файлов контекста (опционально)
    mode: code | research | docs | refactor | auto
    Возвращает список шагов с типом и рекомендуемыми инструментами.
    Используй перед началом сложной многошаговой работы.
    """
    if not task:
        return {"error": "task is required"}

    actual_mode = mode if mode != "auto" else _detect_mode(task)
    steps = _PATTERNS.get(actual_mode, _PATTERNS["code"])

    result = {
        "task": task[:200],
        "detected_mode": actual_mode,
        "steps": steps,
        "total_steps": len(steps),
        "note": "Адаптируй шаги под конкретику задачи; не пропускай verify_result в конце.",
    }

    if context_files:
        result["context_hint"] = f"Начни с чтения: {', '.join(context_files[:5])}"

    return result
