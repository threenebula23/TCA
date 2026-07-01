MODULE: Interface.cli_theme

PURPOSE:
Самостоятельные пресеты цветов для классического CLI (Rich + ANSI).

PUBLIC_API:
|name|description|
|---|---|
|resolve_cli_theme_name|Вернуть id пресета из CLI_THEME_PALETTES.|
|cli_palette|Палитра для Rich/ANSI. Пользовательский акцент меняет только accent, не accent2 —
иначе смена темы визуально «пропадает».|

DEPENDENCIES:
- __future__
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/command_router/_mixin_handlers.py
- Interface/ui_prefs.py
- Interface/visualization.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
