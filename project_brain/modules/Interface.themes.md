# Module: Interface.themes

## Purpose

Движок тем Lorne — 20 тем (10 тёмных + 10 светлых), применяются программно.

---

## Responsibilities

- Движок тем Lorne — 20 тем (10 тёмных + 10 светлых), применяются программно.

---

## Public API

|name|description|
|---|---|
|ensure_custom_textarea_themes|Register all custom TextArea themes once per widget instance.|
|get_theme||
|apply_theme|Применить тему к приложению: CSS-переменные на основных виджетах.|

---

## Dependencies

- `__future__`
- `rich.style`
- `textual._text_area_theme`
- `textual.containers`
- `textual.widgets`
- `typing`
- `ui_prefs`

---

## Used By

- `Interface/modal_style.py`
- `Interface/panels/active_agents_panel.py`
- `Interface/panels/ai_chat/_blocks.py`
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
- `Interface/tui_app.py`
- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/themes.py`

---

## Entry Points


---

## API / route hints

