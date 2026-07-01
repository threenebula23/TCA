# Module: Agent.tools.context7_tool

## Purpose

Инструмент Context7 — прямой вызов REST API.

---

## Responsibilities

- Инструмент Context7 — прямой вызов REST API.

---

## Public API

|name|description|
|---|---|
|resolve_library|Найти библиотеку в Context7 по имени. Возвращает Context7 ID для использования в get_library_docs.
Примеры: 'react', 'fastapi', 'langchain', 'nextjs'.|
|get_library_docs|Получить документацию библиотеки из Context7.
library_id — ID из resolve_library (например '/reactjs/react.dev').
query — что именно нужно найти (например 'hooks useState useEffect').
max_tokens — максимальное число токенов в ответе.|
|get_documentation|Ищет документацию для библиотек и фреймворков.
Если установлен CONTEXT7_API_KEY — использует Context7 API для точных результатов.
Иначе — DuckDuckGo поиск.
query — текст запроса, library — название библиотеки (опционально).|

---

## Dependencies

- `Interface.branding`
- `__future__`
- `ddgs`
- `json`
- `langchain_core.tools`
- `os`
- `typing`
- `urllib.error`
- `urllib.request`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `Agent/tools/compact_tools.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/context7_tool.py`

---

## Entry Points


---

## API / route hints

