# Module: Agent.tools.code_gen

## Purpose

Инструменты генерации кода: запись в файл с правильным расширением.

---

## Responsibilities

- Инструменты генерации кода: запись в файл с правильным расширением.

---

## Public API

|name|description|
|---|---|
|create_code_file|Создаёт/перезаписывает файл с кодом. Если filepath без расширения — добавит расширение по language.
Если filepath оканчивается на .txt, но language указывает на код — заменит на расширение языка.|
|append_code_snippet|Добавляет сниппет кода в конец файла. Если файла нет — создаст. Расширение добавит по language, если его нет.|

---

## Dependencies

- `__future__`
- `langchain_core.tools`
- `path_utils`
- `pathlib`
- `typing`
- `versioning`

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

- `Agent/tools/code_gen.py`

---

## Entry Points


---

## API / route hints

