MODULE: Agent.tools.qa_tool

PURPOSE:
One-shot QA scripts (npm/pnpm) to catch build and framework errors before shipping.

PUBLIC_API:
|name|description|
|---|---|
|run_package_script|npm|pnpm|yarn run <script> (default build) в cwd; для dev — run_command(background=True).|

DEPENDENCIES:
- Agent.path_utils
- Terminal.runner
- __future__
- langchain_core.tools
- path_utils
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
