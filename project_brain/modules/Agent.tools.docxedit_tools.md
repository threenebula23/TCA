# Module: Agent.tools.docxedit_tools

## Purpose

Инструменты правки .docx с сохранением форматирования (docxedit + python-docx).

---

## Responsibilities

- Инструменты правки .docx с сохранением форматирования (docxedit + python-docx).

---

## Public API

|name|description|
|---|---|
|docxedit_replace_keep_format|Заменить все вхождения old_string на new_string в .docx, сохраняя стили (docxedit.replace_string).

Args:
    file_path: Путь к .docx
    old_string: Текст для поиска
    new_string: Замена
    include_tables: учитывать текст в таблицах (ка…|
|docxedit_replace_up_to_paragraph|Замена только до указанного номера абзаца (1-based как в docxedit.replace_string_up_to_paragraph).|
|docxedit_find_line|Показать строку/контекст, где найден search_text (docxedit.show_line).|
|docxedit_table_cell_append|Добавить текст в ячейку таблицы (docxedit.add_text_in_table). Индексы таблицы 0-based; row/column как в docxedit (1-based в API).|
|docxedit_table_font_size|Задать размер шрифта для всей таблицы (pt), docxedit.change_table_font_size.|

---

## Dependencies

- `Agent.path_utils`
- `__future__`
- `docx`
- `docxedit`
- `langchain_core.tools`
- `path_utils`
- `pathlib`
- `typing`

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

- `Agent/tools/docxedit_tools.py`

---

## Entry Points


---

## API / route hints

