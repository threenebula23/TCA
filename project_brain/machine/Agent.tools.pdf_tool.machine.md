MODULE: Agent.tools.pdf_tool

PURPOSE:
Инструмент создания PDF-документа.

PUBLIC_API:
|name|description|
|---|---|
|create_pdf|Создаёт PDF-документ с заданным заголовком и телом. filepath — путь к .pdf файлу.|

DEPENDENCIES:
- langchain_core.tools
- path_utils
- pathlib
- reportlab.lib.pagesizes
- reportlab.lib.units
- reportlab.pdfgen
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/__init__.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
