MODULE: Interface.path_loading

PURPOSE:
Python module `Interface.path_loading`.

PUBLIC_API:
|name|description|
|---|---|
|resolve_path|Разрешает путь (относительный/абсолютный, ~) в абсолютный Path.|
|select_directory|Заглушка: в будущем — диалог выбора директории. Пока возвращает initial или cwd.|

DEPENDENCIES:
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Interface/__init__.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
