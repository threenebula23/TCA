MODULE: Agent.tools.playwright_sync_tool

PURPOSE:
Интерактивная автоматизация сайтов через Playwright Python API (sync).

PUBLIC_API:
|name|description|
|---|---|
|playwright_sync_page_text|Открыть URL и вернуть text_content выбранного элемента (Chromium headless).|
|playwright_sync_click|Перейти на URL, кликнуть по selector, подождать, вернуть URL страницы и заголовок.|
|playwright_sync_fill_and_optional_click|Заполнить поле ввода и опционально нажать кнопку (отправка формы).|
|playwright_sync_screenshot|Скриншот страницы (viewport или full_page).|

DEPENDENCIES:
- Agent.path_utils
- __future__
- json
- langchain_core.tools
- path_utils
- pathlib
- playwright.sync_api
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/compact_tools.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
