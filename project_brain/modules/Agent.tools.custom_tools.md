# Module: Agent.tools.custom_tools

## Purpose

Custom Tools — загрузка, управление и регистрация пользовательских инструментов.

---

## Responsibilities

- Custom Tools — загрузка, управление и регистрация пользовательских инструментов.

---

## Public API

|name|description|
|---|---|
|load_custom_tools|Сканирует каталог кастомных тулов, загружает все @tool-декорированные функции.

Returns:
    Список BaseTool-объектов, готовых к использованию агентом.|
|list_custom_tools|Возвращает информацию обо всех кастомных тулах.

Returns:
    [{name, description, file}, ...]|
|add_custom_tool|Сохраняет кастомный тул как .py файл.

Args:
    name: Имя тула (будет использовано как имя файла без .py)
    code: Python-код с @tool декоратором. Если None — создаётся шаблон.
    description: Описание (для шаблона)

Returns:
    {"ok": …|
|remove_custom_tool|Удаляет кастомный тул.

Args:
    name: Имя тула (без .py)

Returns:
    {"ok": True} или {"ok": False, "error": str}|
|get_custom_tools_prompt|Формирует блок для system prompt с описанием кастомных тулов.

Returns:
    Строка с описанием, или пустая строка если тулов нет.|
|reload_custom_tools|Перезагрузить все кастомные тулы (очистить кэш модулей и загрузить заново).|

---

## Dependencies

- `Agent.runtime_paths`
- `Interface.visualization`
- `__future__`
- `importlib.util`
- `langchain_core.tools`
- `os`
- `pathlib`
- `sys`
- `traceback`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/custom_tools.py`

---

## Entry Points


---

## API / route hints

