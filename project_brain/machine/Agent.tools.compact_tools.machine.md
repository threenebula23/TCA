MODULE: Agent.tools.compact_tools

PURPOSE:
Компактные мульти-тулы: одна схема вместо нескольких — меньше токенов на bind_tools.

PUBLIC_API:
|name|description|
|---|---|
|plan_tool|План: save (title + steps или steps_json), load, update (step_index, status), clear.

save: предпочти массив `steps`; иначе `steps_json` — одна строка, полный JSON `["a","b"]`.|
|docx_write_tool|Создать/дописать/патчить .docx.

action=create|append: data_json как paragraphs_json;
action=patch: data_json как patches_json.|
|docxedit_tool|Правки .docx с сохранением формата.

action=replace|replace_limited: old_string, new_string (replace_limited + paragraph_number).
action=find_line: search_text. action=table_cell: table_index, row_num, column_num, new_string.
action=table_f…|
|ocr_tool|OCR / чтение текста.

action=soft: .txt/.md/.py/.pdf (текстовый слой). action=medium: скрин/UI.
action=strong: фото. Для всех: path, max_chars; для soft — max_pdf_pages;
для medium/strong — max_side.|
|code_file_tool|Создать файл (action=create: filepath, language, code) или дописать фрагмент (action=append: filepath, snippet).|
|git_ops|Git: status | log (path, limit) | diff (commit, пусто=незакоммиченное) | rollback_file (path, commit).|
|library_context|Context7: resolve (library_name) | docs (library_id, query) | search (query, опц. library_name).|
|headless_browser|Node Chromium: get_text|screenshot|click_and_get|evaluate; url/selector/wait_ms по схеме.|
|playwright_sync|Python Playwright (sync), только при включении в Settings режима Agent.

action=page_text: url, selector. action=click: url, click_selector, wait_after_ms.
action=fill_submit: url, field_selector, fill_text, button_selector (опц.).
action=s…|
|reasoning_tool|Рассуждения и анализ.

action=think: короткая запись в thought. action=diff: path, old_content, new_content
(unified diff перед edit). action=analyze: path, query (RAG по файлу).|
|project_brain_tool|Brain: refresh|reindex|scan; write_architecture; write_brain (brain_rel_path + content).|
|file_versions_tool|Версии файла: list | rollback.|

DEPENDENCIES:
- Agent.path_utils
- Agent.project_brain
- Agent.project_brain.agent_architecture
- Agent.rag
- Agent.tools.browser_tool
- Agent.tools.code_gen
- Agent.tools.context7_tool
- Agent.tools.docxedit_tools
- Agent.tools.git_tool
- Agent.tools.ocr_tool
- Agent.tools.office_document_tool
- Agent.tools.planning_tool
- Agent.tools.playwright_sync_tool
- Agent.tools.thinking_tool
- Agent.tools.versioning_tool
- __future__
- browser_tool
- code_gen
- context7_tool
- docxedit_tools
- git_tool
- json
- json_repair
- langchain_core.tools
- ocr_tool
- office_document_tool
- planning_tool
- playwright_sync_tool
- thinking_tool
- typing
- versioning_tool

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py
- tests/test_project_brain_tool.py

RISKS:
