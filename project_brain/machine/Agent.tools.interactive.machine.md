MODULE: Agent.tools.interactive

PURPOSE:
Интерактивные инструменты: запрос ввода у пользователя в терминале.

PUBLIC_API:
|name|description|
|---|---|
|ask_user|Спросить пользователя в терминале. Выводит question и возвращает ответ пользователя. Используй для подтверждения действий (например, запуск команды), выбора варианта или уточнения.|

DEPENDENCIES:
- Interface.tui_bridge
- langchain_core.tools
- sys
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
