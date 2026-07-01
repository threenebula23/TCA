MODULE: Agent.tools.browser_tool

PURPOSE:
Browser automation tool using Playwright CLI for agent mode.

PUBLIC_API:
|name|description|
|---|---|
|browser_get_text|Fetch text content from a web page using a headless browser.

Args:
    url: The URL to navigate to.
    selector: CSS selector to extract text from (default: 'body').
    wait_ms: Milliseconds to wait for the page to load.|
|browser_screenshot|Take a screenshot of a web page.

Args:
    url: The URL to screenshot.
    output_path: Where to save the screenshot.
    full_page: Whether to capture the full page or just the viewport.|
|browser_click_and_get|Navigate to a page, click an element, then extract text.

Args:
    url: The URL to navigate to.
    click_selector: CSS selector of the element to click.
    result_selector: CSS selector to extract text from after clicking.
    wait_ms: M…|
|browser_evaluate|Navigate to a page and evaluate a JavaScript expression.

Args:
    url: The URL to navigate to.
    js_expression: JavaScript expression to evaluate in the page context.|

DEPENDENCIES:
- __future__
- json
- langchain_core.tools
- pathlib
- subprocess
- tempfile
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/__init__.py
- Agent/tools/compact_tools.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
