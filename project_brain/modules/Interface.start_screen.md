# Module: Interface.start_screen

## Purpose

Initial TUI screen for selecting a project before launching the main app.

---

## Responsibilities

- Initial TUI screen for selecting a project before launching the main app.

---

## Public API

|name|description|
|---|---|
|load_recent_projects||
|save_recent_project||
|select_project_path||

---

## Dependencies

- `Agent.runtime_paths`
- `Interface.branding`
- `Interface.modal_style`
- `Interface.ui_prefs`
- `__future__`
- `collections`
- `json`
- `math`
- `pathlib`
- `pyfiglet`
- `rich.align`
- `rich.style`
- `rich.text`
- `textual`
- `textual.app`
- `textual.containers`
- `textual.screen`
- `textual.widget`
- `textual.widgets`
- `typing`

---

## Used By

- `Agent/agent/_impl_tui.py`
- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/start_screen.py`

---

## Entry Points


---

## API / route hints

