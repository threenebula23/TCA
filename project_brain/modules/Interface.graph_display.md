# Module: Interface.graph_display

## Purpose

Graph Display — Rich-визуализация работы параллельных агентов Creator Mode.

---

## Responsibilities

- Graph Display — Rich-визуализация работы параллельных агентов Creator Mode.

---

## Public API

|name|description|
|---|---|
|pause_live_display|Пауза Live-отображения для ввода (с блокировкой).|
|build_graph_renderable|Построить Rich-рендеринг графа агентов.

Returns:
    Panel с визуализацией (для использования в Live)|
|display_creator_result|Показать финальный результат Creator Mode.|

---

## Dependencies

- `__future__`
- `contextlib`
- `rich`
- `rich.columns`
- `rich.console`
- `rich.layout`
- `rich.live`
- `rich.panel`
- `rich.table`
- `rich.text`
- `sys`
- `threading`
- `time`
- `typing`

---

## Used By

- `Agent/creator_mode.py`
- `Agent/tools/terminal_tool.py`
- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/graph_display.py`

---

## Entry Points


---

## API / route hints

