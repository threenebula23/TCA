# Module: Agent.tools.file_ops

## Purpose

Инструменты работы с файлами: чтение, листинг, поиск в подпапках, редактирование.

---

## Responsibilities

- Инструменты работы с файлами: чтение, листинг, поиск в подпапках, редактирование.

---

## Public API

|name|description|
|---|---|
|read_file|Чтение файла: offset/limit (0-based строки); без диапазона длинные файлы обрезаются (~400 строк) — дальше read_file_lines.|
|read_file_lines|Фрагмент по строкам (1-based start_line..end_line; end_line=0 — до конца, кап 5000 строк). Номера в content как ``N|``.|
|list_files|Список файлов в каталоге. path — папка ("" или "." = текущая). recursive — рекурсивно. pattern — glob по **имени** файла (например *.py).|
|search_in_files|Поиск текста в файлах в директории и подпапках. directory — корень поиска, query — строка для поиска, file_pattern — glob (например *.py).|
|find_in_file|Один файл: подстрока или regex + номера строк; большие файлы — сначала read_file_lines.|
|edit_file|Заменяет первое вхождение old_str на new_str. Пустой old_str — перезапись файла содержимым new_str.|
|write_file|Создать или перезаписать файл содержимым (content).|
|replace_file_lines|Заменить строки start_line..end_line (1-based, включительно) на content. Пустой content — удаление диапазона.|
|insert_file_lines|Вставить content после строки after_line (0 = в начало; k = после k-й строки, 1-based).|
|get_file_line_count|Возвращает количество строк в файле. Полезно для отображения состояния файла.|

---

## Dependencies

- `Agent.git_integration`
- `fnmatch`
- `langchain_core.tools`
- `os`
- `path_utils`
- `pathlib`
- `re`
- `typing`
- `versioning`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `Agent/tools/thinking_tool.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/file_ops.py`

---

## Entry Points


---

## API / route hints

