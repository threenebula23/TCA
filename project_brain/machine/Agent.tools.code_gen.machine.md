MODULE: Agent.tools.code_gen

PURPOSE:
Инструменты генерации кода: запись в файл с правильным расширением.

PUBLIC_API:
|name|description|
|---|---|
|create_code_file|Создаёт/перезаписывает файл с кодом. Если filepath без расширения — добавит расширение по language.
Если filepath оканчивается на .txt, но language указывает на код — заменит на расширение языка.|
|append_code_snippet|Добавляет сниппет кода в конец файла. Если файла нет — создаст. Расширение добавит по language, если его нет.|

DEPENDENCIES:
- __future__
- langchain_core.tools
- path_utils
- pathlib
- typing
- versioning

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/__init__.py
- Agent/tools/compact_tools.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
