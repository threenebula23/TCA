# Module: Interface.cli_theme

## Purpose

Самостоятельные пресеты цветов для классического CLI (Rich + ANSI).

---

## Responsibilities

- Самостоятельные пресеты цветов для классического CLI (Rich + ANSI).

---

## Public API

|name|description|
|---|---|
|resolve_cli_theme_name|Вернуть id пресета из CLI_THEME_PALETTES.|
|cli_palette|Палитра для Rich/ANSI. Пользовательский акцент меняет только accent, не accent2 —
иначе смена темы визуально «пропадает».|

---

## Dependencies

- `__future__`
- `typing`

---

## Used By

- `Agent/command_router/_mixin_handlers.py`
- `Interface/ui_prefs.py`
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

- `Interface/cli_theme.py`

---

## Entry Points


---

## API / route hints

