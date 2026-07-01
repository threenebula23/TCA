# Module: Agent.tools.office_document_tool

## Purpose

Чтение и запись Office/PDF: DOCX со стилями Word, простой PDF с типографикой ReportLab, .doc через antiword (если есть).

---

## Responsibilities

- Чтение и запись Office/PDF: DOCX со стилями Word, простой PDF с типографикой ReportLab, .doc через antiword (если есть).

---

## Public API

|name|description|
|---|---|
|office_document_read|Читает .docx (абзацы + имена стилей Word), .pdf (текст по страницам) или .doc (только текст, нужен antiword).
Для правок стилей и текста используй docx_document_patch_paragraphs / docx_document_append_paragraphs.|
|docx_document_create|Создаёт новый .docx. paragraphs_json: [{"text":"...","style":"Title|Heading 1|Normal|..."}, ...].|
|docx_document_append_paragraphs|Добавляет в конец существующего .docx абзацы. paragraphs_json как в docx_document_create.|
|docx_document_patch_paragraphs|Правка абзацев по индексу (0-based). patches_json:
[{"paragraph_index": 0, "text": "новый текст", "style": "Heading 2"}, ...]
Поле style можно опустить — останется прежний стиль (кроме смены через text).|
|docx_document_advanced_ops|Массив op в operations_json (до 40): append_paragraph, set_paragraph_*, set_run_font, set_section_*, insert_page_break_after_paragraph, insert_table_after_paragraph. Детали полей — из ответа валидации; TOC/PAGE — через code_interpreter.|
|pdf_styled_document_create|PDF через ReportLab; sections_json: [{role: title|h1|h2|body, text}, ...]; перезапись файла.|

---

## Dependencies

- `Agent.path_utils`
- `__future__`
- `docx`
- `docx.enum.section`
- `docx.enum.text`
- `docx.shared`
- `fitz`
- `json`
- `langchain_core.tools`
- `path_utils`
- `pathlib`
- `re`
- `reportlab.lib.pagesizes`
- `reportlab.lib.styles`
- `reportlab.lib.units`
- `reportlab.platypus`
- `subprocess`
- `typing`
- `xml.sax.saxutils`

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

- `Agent/tools/office_document_tool.py`

---

## Entry Points


---

## API / route hints

