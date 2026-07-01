# Module: Interface.ui_prefs

## Purpose

Настройки UI (тема, плотность, синтаксис) в каталоге данных проекта (``.lorne`` / legacy ``.tca``).

---

## Responsibilities

- Настройки UI (тема, плотность, синтаксис) в каталоге данных проекта (``.lorne`` / legacy ``.tca``).

---

## Public API

|name|description|
|---|---|
|prefs_path||
|load_prefs||
|cli_prompt_prefix_plain|Plain-text CLI prompt prefix (safe for Rich); ends with a space.|
|save_prefs||

---

## Dependencies

- `Agent.path_utils`
- `Agent.runtime_paths`
- `Interface.cli_theme`
- `__future__`
- `json`
- `pathlib`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/command_router/_main.py`
- `Agent/command_router/_mixin_handlers.py`
- `Agent/creator_provider.py`
- `Agent/tools/parallel_helper_tool.py`
- `Interface/input_widget.py`
- `Interface/modal_style.py`
- `Interface/panels/active_agents_panel.py`
- `Interface/panels/ai_chat/_blocks.py`
- `Interface/panels/ai_chat/_helpers.py`
- `Interface/panels/ai_chat/_mixin_events.py`
- `Interface/panels/ai_chat/_mixin_setup.py`
- `Interface/panels/ai_chat/_mixin_stream.py`
- `Interface/panels/code_editor/_part_a.py`
- `Interface/panels/code_editor/_part_b.py`
- `Interface/panels/creator_progress.py`
- `Interface/panels/deep_checkpoint.py`
- `Interface/panels/diff_block.py`
- `Interface/panels/download_block.py`
- `Interface/panels/file_explorer.py`
- `Interface/panels/thinking_block.py`
- `Interface/panels/tool_card.py`
- `Interface/panels/usage_calendar.py`
- `Interface/panels/version_control.py`
- `Interface/start_screen.py`
- `Interface/tui_app.py`
- `Interface/visualization.py`
- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/ui_prefs.py`

---

## Entry Points


---

## API / route hints

