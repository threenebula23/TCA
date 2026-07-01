# Module: Interface.branding

## Purpose

Константы продукта Lorne: имя, версия, вспомогательные строки для UI и HTTP.

---

## Responsibilities

- Константы продукта Lorne: имя, версия, вспомогательные строки для UI и HTTP.

---

## Public API

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

---

## Dependencies

- `__future__`

---

## Used By

- `Agent/tools/context7_tool.py`
- `Agent/tools/download_tool.py`
- `Agent/tools/web_tool.py`
- `Interface/panels/ai_chat/_mixin_setup.py`
- `Interface/splash.py`
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

- `Interface/branding.py`

---

## Entry Points


---

## API / route hints

