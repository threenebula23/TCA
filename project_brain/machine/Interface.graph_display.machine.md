MODULE: Interface.graph_display

PURPOSE:
Graph Display — Rich-визуализация работы параллельных агентов Creator Mode.

PUBLIC_API:
|name|description|
|---|---|
|pause_live_display|Пауза Live-отображения для ввода (с блокировкой).|
|build_graph_renderable|Построить Rich-рендеринг графа агентов.

Returns:
    Panel с визуализацией (для использования в Live)|
|display_creator_result|Показать финальный результат Creator Mode.|

DEPENDENCIES:
- __future__
- contextlib
- rich
- rich.columns
- rich.console
- rich.layout
- rich.live
- rich.panel
- rich.table
- rich.text
- sys
- threading
- time
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/creator_mode.py
- Agent/tools/terminal_tool.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
