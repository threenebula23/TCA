MODULE: Agent.file_loading

PURPOSE:
Python module `Agent.file_loading`.

PUBLIC_API:
|name|description|
|---|---|
|load_file|Загружает содержимое файла как текст.|
|load_directory_texts|Загружает текстовое содержимое файлов в директории (и подпапках). Возвращает [(путь, текст), ...].|

DEPENDENCIES:
- path_utils
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/rag/__init__.py
- tests/test_ollama_provider.py

RISKS:
