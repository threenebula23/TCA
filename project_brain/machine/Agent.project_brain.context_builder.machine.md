MODULE: Agent.project_brain.context_builder

PURPOSE:
Normalize ``scan_project`` output into a Relator-ready context (flat lists + nested dicts).

PUBLIC_API:
|name|description|
|---|---|
|build_project_context|Build the JSON model consumed by Relator templates and ``rag_manifest``.|

DEPENDENCIES:
- Agent.path_utils
- __future__
- collections
- datetime
- json
- pathlib
- re
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- Agent/graph_runner.py
- Agent/tools/compact_tools.py
- tests/test_ollama_provider.py
- tests/test_project_brain_tool.py

RISKS:
