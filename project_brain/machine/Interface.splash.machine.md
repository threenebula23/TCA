MODULE: Interface.splash

PURPOSE:
Краткий splash при старте (Rich + pyfiglet) для Lorne.

PUBLIC_API:
|name|description|
|---|---|
|show_splash|Печатает баннер с именем продукта и версией (до поднятия Textual).|

DEPENDENCIES:
- Interface.branding
- __future__
- pyfiglet
- rich
- rich.console
- rich.panel
- rich.text
- sys

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
