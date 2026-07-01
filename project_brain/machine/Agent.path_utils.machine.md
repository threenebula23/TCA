MODULE: Agent.path_utils

PURPOSE:
Общая утилита разрешения путей для агента и модуля загрузки Path.

PUBLIC_API:
|name|description|
|---|---|
|set_project_root||
|clear_project_root|Reset project root override (tests / subprocess boundaries).|
|get_project_root|Текущий корень проекта (как у resolve_abs_path для относительных путей).|
|resolve_abs_path||

DEPENDENCIES:
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- Agent/checkpoint/__init__.py
- Agent/creator_mode.py
- Agent/deep_solver/legacy_loop.py
- Agent/graph_runner.py
- Agent/project_brain/__init__.py
- Agent/project_brain/agent_architecture.py
- Agent/project_brain/context_builder.py
- Agent/rag/__init__.py
- Agent/tools/compact_tools.py
- Agent/tools/docxedit_tools.py
- Agent/tools/ocr_tool.py
- Agent/tools/office_document_tool.py
- Agent/tools/playwright_sync_tool.py
- Agent/tools/qa_tool.py
- Agent/tools/versioning_tool.py
- Interface/ui_prefs.py
- Terminal/runner.py
- tests/test_checkpoint_rollback.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py
- tests/test_project_brain_tool.py

RISKS:
