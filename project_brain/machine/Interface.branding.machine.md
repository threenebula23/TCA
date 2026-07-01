MODULE: Interface.branding

PURPOSE:
Константы продукта Lorne: имя, версия, вспомогательные строки для UI и HTTP.

PUBLIC_API:
|name|description|
|---|---|
|cli_attractor_block|Статичный мини-блок слева от логотипа в classic CLI (палитра фиолетового TUI).

Возвращает многострочную ASCII-«искру» без анимации — только визуальный якорь.

Пример::

    from rich.text import Text
    from Interface.branding import cli_…|
|user_agent_fragment|Фрагмент User-Agent для исходящих HTTP-запросов (совместимость с логами серверов).

Пример::

    headers = {"User-Agent": f"Mozilla/5.0 (compatible; {user_agent_fragment()})"}|

DEPENDENCIES:
- __future__

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/tools/context7_tool.py
- Agent/tools/download_tool.py
- Agent/tools/web_tool.py
- Interface/panels/ai_chat/_mixin_setup.py
- Interface/splash.py
- Interface/start_screen.py
- Interface/tui_app.py
- Interface/visualization.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
