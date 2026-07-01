MODULE: Agent.runtime_paths

PURPOSE:
Каталоги данных и env: префикс ``LORNE_*`` / ``.lorne`` с откатом на ``TCA_*`` / ``.tca``.

PUBLIC_API:
|name|description|
|---|---|
|env_pref|Читает ``LORNE_<suffix>``, затем ``TCA_<suffix>`` (совместимость).|
|project_data_dir|Каталог данных в проекте: ``.lorne`` приоритетно; иначе существующий ``.tca``.|
|user_config_json_path|Глобальный JSON-конфиг: ``~/.lorne_config.json`` или legacy ``~/.tca_config.json``.|
|custom_tools_dir|Каталог кастомных тулов: ``~/.lorne_custom_tools`` или legacy.|
|recent_projects_json_path|Недавние проекты: ``~/.lorne_recent_projects.json`` или legacy.|

DEPENDENCIES:
- __future__
- os
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- Agent/checkpoint/__init__.py
- Agent/deep_solver/_impl_a.py
- Agent/deep_solver/legacy_loop.py
- Agent/llm_provider.py
- Agent/rag/__init__.py
- Agent/tools/custom_tools.py
- Agent/tools/planning_tool.py
- Agent/tools/terminal_tool.py
- Agent/versioning/__init__.py
- Interface/panels/ai_chat/_mixin_setup.py
- Interface/panels/usage_calendar.py
- Interface/start_screen.py
- Interface/ui_prefs.py
- Terminal/runner.py
- tests/test_ollama_provider.py

RISKS:
