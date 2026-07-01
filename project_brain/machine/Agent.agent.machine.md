MODULE: Agent.agent

PURPOSE:
Агент Lorne — пакет (раньше один модуль ``agent.py``, дубль удалён).

PUBLIC_API:


DEPENDENCIES:
- __future__
- _impl_classic
- _impl_prepare
- _impl_tui

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_tui.py
- Agent/command_router/_main.py
- Interface/panels/file_explorer.py
- lorne.py
- tests/test_ollama_provider.py
- tests/test_prompt_budget.py

RISKS:
