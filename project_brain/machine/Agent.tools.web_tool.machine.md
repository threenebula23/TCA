MODULE: Agent.tools.web_tool

PURPOSE:
Инструменты web_search / web_fetch для агента Lorne.

PUBLIC_API:
|name|description|
|---|---|
|web_search|Поиск в интернете: короткие сниппеты и список URL. Для полного текста страницы вызови web_fetch(url).
max_results — не больше 8; snippet_chars — длина сниппета (экономия токенов).|
|web_fetch|Загружает одну страницу: текст сжат до max_length (умная обрезка). code_block_chars — лимит на блок кода.
Предпочитай web_search → несколько узких web_fetch вместо web_search_and_read на большие объёмы.|
|web_search_and_read|Поиск + чтение первых страниц. По умолчанию мало страниц и короткий текст — экономия токенов.
Для глубины лучше: web_search → web_fetch по выбранным URL.|

DEPENDENCIES:
- Agent.config
- Interface.branding
- __future__
- ddgs
- hashlib
- html
- langchain_core.tools
- re
- time
- typing
- urllib.error
- urllib.request

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/__init__.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
