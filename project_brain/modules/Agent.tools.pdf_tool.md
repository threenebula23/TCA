# Module: Agent.tools.pdf_tool

## Purpose

Инструмент создания PDF-документа.

---

## Responsibilities

- Инструмент создания PDF-документа.

---

## Public API

|name|description|
|---|---|
|create_pdf|Создаёт PDF-документ с заданным заголовком и телом. filepath — путь к .pdf файлу.|

---

## Dependencies

- `langchain_core.tools`
- `path_utils`
- `pathlib`
- `reportlab.lib.pagesizes`
- `reportlab.lib.units`
- `reportlab.pdfgen`
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

- `Agent/tools/pdf_tool.py`

---

## Entry Points


---

## API / route hints

