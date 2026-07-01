MODULE: Agent.message_utils

PURPOSE:
Утилиты сообщений LLM: нормализация вызовов инструментов, компактность, восстановление JSON.

PUBLIC_API:


DEPENDENCIES:
- __future__
- _impl_high
- _impl_low

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- Agent/background_agent_runner.py
- Agent/checkpoint/__init__.py
- Agent/command_router/_main.py
- Agent/creator_mode.py
- Agent/deep_solver/_impl_a.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Interface/panels/ai_chat/_helpers.py
- Interface/panels/ai_chat/_mixin_stream.py
- tests/test_message_utils_tools.py
- tests/test_ollama_provider.py
- tests/test_package_imports.py

RISKS:
