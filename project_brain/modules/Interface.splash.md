# Module: Interface.splash

## Purpose

Краткий splash при старте (Rich + pyfiglet) для Lorne.

---

## Responsibilities

- Краткий splash при старте (Rich + pyfiglet) для Lorne.

---

## Public API

|name|description|
|---|---|
|show_splash|Печатает баннер с именем продукта и версией (до поднятия Textual).|

---

## Dependencies

- `Interface.branding`
- `__future__`
- `pyfiglet`
- `rich`
- `rich.console`
- `rich.panel`
- `rich.text`
- `sys`

---

## Used By

- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/splash.py`

---

## Entry Points


---

## API / route hints

