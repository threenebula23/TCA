MODULE: Interface.visualization

PURPOSE:
Красивый терминальный вывод для агента Lorne (classic CLI) — вдохновлён Claude Code.

PUBLIC_API:
|name|description|
|---|---|
|refresh_cli_ui_from_prefs|Перечитать cli_theme и cli_accent_color; обновить Rich Console и ANSI для plain-режима.|
|get_context_limit||
|set_cli_progress_round||
|set_cli_progress_plan_step||
|section||
|step||
|round_header||
|display_agent_action||
|display_tool_result||
|display_model_reply||
|display_turn_summary||
|display_usage||
|display_cumulative_usage||
|print_startup_banner|Стартовый баннер classic CLI: аттрактор слева, figlet имени, метаданные сессии.|
|print_welcome|Обратная совместимость: тот же вид, что стартовый баннер (без дублирования логики).|
|display_shell_command|Display a user-initiated shell command with terminal-like styling.|
|display_model_selector|Display a rich model selection interface grouped by tier.|
|display_status_panel|Display a formatted status panel for /status command.|
|print_help_topic|Показать строки справки, где встречается topic (команда или описание).|
|print_commands||
|print_session_list||
|print_thinking||
|print_planning||
|print_deep_cli_session_banner|Rich-блок: что Deep Solver делает и что можно вводить (классический CLI).|
|print_deep_cli_heartbeat|Компактная строка: время, чекпоинты, шаг (throttle снаружи).

Для classic CLI ``to_stderr=True`` — не смешивать с stdout/prompt_toolkit.|
|print_deep_cli_checkpoint||
|print_info_block|Несколько строк в одной Rich-панели (команды /mode, /ollama, …).|
|print_info||
|print_success||
|print_warning||
|print_error||
|get_user_input||
|read_cli_line|Одна строка ввода; при EOF/Ctrl+D — пустая строка (не ``/exit``), чтобы не путать с id модели.|
|display_file_diffs|Визуализация списка измененных файлов (как в обычном агенте).|
|display_rag_progress|Display RAG indexing progress inline.|
|display_rag_results|Display RAG search results in a formatted table.|
|display_enhanced_status|Enhanced status panel with RAG, versioning, and creator mode info.|
|suggest_command|Suggest a command if user input looks like a mistyped command.|

DEPENDENCIES:
- Agent.llm_provider
- Interface.branding
- Interface.cli_theme
- Interface.tui_bridge
- Interface.ui_prefs
- __future__
- io
- json
- pathlib
- plotext
- pyfiglet
- re
- rich
- rich.columns
- rich.console
- rich.live
- rich.markdown
- rich.panel
- rich.progress
- rich.rule
- rich.spinner
- rich.syntax
- rich.table
- rich.text
- rich.theme
- sys
- time
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/command_router/_main.py
- Agent/command_router/_mixin_handlers.py
- Agent/creator_mode.py
- Agent/deep_solver/legacy_loop.py
- Agent/graph_runner.py
- Agent/message_utils/_impl_high.py
- Agent/message_utils/_impl_low.py
- Agent/spinner.py
- Agent/tools/custom_tools.py
- Interface/__init__.py
- Interface/panels/usage_calendar.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
