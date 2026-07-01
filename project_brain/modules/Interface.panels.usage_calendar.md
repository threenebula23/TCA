# Module: Interface.panels.usage_calendar

## Purpose

GitHub-style usage-calendar widget for OpenRouter balance display.

---

## Responsibilities

- GitHub-style usage-calendar widget for OpenRouter balance display.

---

## Public API

|name|description|
|---|---|
|record_cumulative_usage|Persist today's usage delta given the cumulative total reported by OpenRouter.

Returns the updated log. This function is idempotent within a day —
calling it multiple times just keeps the **maximum** delta we've seen
so far, which correspo…|
|total_usage||
|render_cli_usage_calendar_text|Календарь расходов OpenRouter для Rich Panel в CLI (те же данные, что в TUI).|

---

## Dependencies

- `Agent.runtime_paths`
- `Interface.themes`
- `Interface.ui_prefs`
- `Interface.visualization`
- `__future__`
- `datetime`
- `json`
- `math`
- `pathlib`
- `rich.text`
- `textual.app`
- `textual.containers`
- `textual.widgets`
- `typing`

---

## Used By

- `Agent/command_router/_main.py`
- `Interface/panels/ai_chat/_mixin_events.py`
- `Interface/panels/ai_chat/_mixin_setup.py`
- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/panels/usage_calendar.py`

---

## Entry Points


---

## API / route hints

