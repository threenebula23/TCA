# Module: tests.test_latest_fixes

## Purpose

Tests for the latest round of fixes: plan_tool coercion, download_file,

---

## Responsibilities

- Tests for the latest round of fixes: plan_tool coercion, download_file,

---

## Public API

|name|description|
|---|---|
|test_plan_tool_accepts_uppercase_action||
|test_plan_tool_coerces_native_steps_list||
|test_plan_tool_status_synonyms||
|test_plan_tool_action_falls_back_to_save_or_load_for_unknown||
|test_plan_tool_truncates_oversize_strings||
|test_download_file_requires_full_url||
|test_download_tool_registered||
|test_download_cancel_flag_lifecycle||
|test_run_command_card_shows_command_and_elapsed||
|test_read_file_card_timer_present_when_stamped||
|test_write_file_accepts_body_and_target||
|test_write_file_content_from_list||
|test_code_file_tool_action_write_maps_to_create||
|test_code_file_tool_snippet_promoted_to_code_on_create||
|test_code_file_tool_append_fills_snippet_from_code||
|test_list_files_accepts_directory_alias||
|test_run_command_accepts_cmd_alias||
|test_replace_file_lines_coerces_string_line_numbers||
|test_terminal_tool_stamps_elapsed_seconds||

---

## Dependencies

- `Agent.tool_registry`
- `Agent.tool_schemas`
- `Agent.tools.download_tool`
- `Agent.tools.terminal_tool`
- `Interface.panels.tool_card`
- `Terminal.runner`
- `__future__`

---

## Used By


---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `tests/test_latest_fixes.py`

---

## Entry Points


---

## API / route hints

