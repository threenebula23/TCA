MODULE: Agent.creator_summary

PURPOSE:
Единый текст итога Creator Mode для TUI, classic CLI и истории сообщений.

PUBLIC_API:
|name|description|
|---|---|
|format_creator_summary_text|Markdown: статус, все воркеры, полный result (с защитными лимитами по длине).|

DEPENDENCIES:
- __future__
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/command_router/_main.py
- Agent/command_router/_mixin_handlers.py
- Agent/deep_solver/legacy_loop.py
- tests/test_ollama_provider.py

RISKS:
