# Сводная таблица инструментов

Источник истины по составу: `Agent/tool_registry.py` (`_base_tools`, `_extended_tools`, `build_tools`, `_ASK_EXCLUDED_TOOL_NAMES`, `_CUSTOM_TOOL_NAMES`, `_EXTENDED_TOOL_NAMES`).

Легенда:

- **Base** — всегда в сессии при включённых custom tools (кроме фильтра Ask).
- **+Git** — `git_ops`, если импорт `git_tool` успешен.
- **+Agent** — добавляется в режимах с `agent_mode=True` при prefs: `headless_browser` (если `browser_tools_enabled`), `playwright_sync` (если `playwright_python_enabled`).

## Имя у модели → реализация

| Имя | Где реализовано |
|-----|-----------------|
| `read_file` | `Agent/tools/file_ops.py` |
| `read_file_lines` | `Agent/tools/file_ops.py` |
| `list_files` | `Agent/tools/file_ops.py` |
| `edit_file` | `Agent/tools/file_ops.py` |
| `write_file` | `Agent/tools/file_ops.py` |
| `replace_file_lines` | `Agent/tools/file_ops.py` |
| `insert_file_lines` | `Agent/tools/file_ops.py` |
| `get_file_line_count` | `Agent/tools/file_ops.py` |
| `code_file_tool` | `Agent/tools/compact_tools.py` → `code_gen` |
| `plan_tool` | `Agent/tools/compact_tools.py` → `planning_tool` |
| `search_in_files` | `Agent/tools/file_ops.py` |
| `find_in_file` | `Agent/tools/file_ops.py` |
| `run_command` | `Agent/tools/terminal_tool.py` |
| `run_package_script` | `Agent/tools/qa_tool.py` |
| `download_file` | `Agent/tools/download_tool.py` |
| `create_pdf` | `Agent/tools/pdf_tool.py` |
| `ask_user` | `Agent/tools/interactive.py` |
| `web_search` | `Agent/tools/web_tool.py` |
| `web_fetch` | `Agent/tools/web_tool.py` |
| `start_background_task` | `Agent/tools/parallel_helper_tool.py` |
| `get_background_result` | `Agent/tools/parallel_helper_tool.py` |
| `ocr_tool` | `Agent/tools/compact_tools.py` |
| `office_document_read` | `Agent/tools/office_document_tool.py` |
| `docx_write_tool` | `Agent/tools/compact_tools.py` |
| `docx_document_advanced_ops` | `Agent/tools/office_document_tool.py` |
| `docxedit_tool` | `Agent/tools/compact_tools.py` |
| `pdf_styled_document_create` | `Agent/tools/office_document_tool.py` |
| `reasoning_tool` | `Agent/tools/compact_tools.py` |
| `code_interpreter` | `Agent/tools/code_interpreter.py` |
| `rag_search` | `Agent/rag/__init__.py` (`get_rag_tool`) |
| `project_brain_tool` | `Agent/tools/compact_tools.py` |
| `git_ops` | `Agent/tools/compact_tools.py` → `git_tool` (опционально) |
| `library_context` | `Agent/tools/compact_tools.py` → `context7_tool` |
| `file_versions_tool` | `Agent/tools/compact_tools.py` |
| `headless_browser` | `Agent/tools/compact_tools.py` (+Agent, prefs) |
| `playwright_sync` | `Agent/tools/compact_tools.py` (+Agent, prefs) |
| `spawn_subagent` | `Agent/tools/subagent_tools.py` → `Agent/subagent_runner.py` (не в Ask) |
| `get_subagent_result` | `Agent/tools/subagent_tools.py` → `Agent/subagent_runner.py` (не в Ask) |

## Базовый тир для слабых моделей

| Имя | Где реализовано |
|-----|-----------------|
| `structured_memory` | `Agent/tools/memory_tool.py` |
| `ast_analyze` | `Agent/tools/ast_tool.py` |
| `multi_read` | `Agent/tools/multi_read_tool.py` |
| `lint_check` | `Agent/tools/lint_tool.py` |
| `task_decompose` | `Agent/tools/decompose_tool.py` |
| `env_info` | `Agent/tools/env_tool.py` |
| `batch_replace` | `Agent/tools/batch_replace_tool.py` |
| `verify_result` | `Agent/tools/verify_tool.py` |
| `session_notes` | `Agent/tools/notes_tool.py` |

## Расширенный тир (опционально, `extended_tools_enabled`)

Мега-тулы из **`Agent/tools/extended_tools.py`** — включаются тумблером **Extended tools** в настройках (по умолчанию выключен, см. `Interface/ui_prefs.py`). Часть базовых тулов (`ast_analyze`, `search_in_files`, `lint_check`, `session_notes`) заменяется соответствующими действиями мега-тулов, когда расширенный тир включён (`_EXTENDED_REPLACES` в `tool_registry.py`).

| Имя | Действия / назначение |
|-----|------------------------|
| `code_intel_tool` | Код/структура (замена `ast_analyze`) |
| `workspace_search` | Поиск по репозиторию (замена `search_in_files`) |
| `net_tool` | `http` \| `port_check` \| `db_query` (SQLite, только `SELECT`/`PRAGMA table_info`) |
| `viz_tool` | `chart` \| `diagram` — данные для рендера в чате (см. [Interface/OVERVIEW.md](Interface/OVERVIEW.md)) |
| `qa_extended_tool` | `test` (pytest/`run_command`) \| `lint` (замена `lint_check`) |
| `session_meta_tool` | `transcript_search` \| `config_inspect` \| `json_validate` \| `review_checklist` \| `notify_when_done` |
| `tools_catalog` | `list` \| `describe` — справочник по тулам по требованию |
| `diff_tool` | Git diff по коммиту/пути |
| `apply_patch` | Применить unified diff |
| `project_tree` | Дерево проекта |
| `brain_search` | Поиск по Project Brain |
| `export_to_brain` | Экспорт заметок в `project_brain/agent/*.md` |
| `memory_search` | Поиск по `structured_memory` |

Пользовательские туловые модули добавляются через `custom_tools` (см. [tool/REFERENCE.md](tool/REFERENCE.md#custom-tools)).

Детали аргументов и примеров: [tool/REFERENCE.md](tool/REFERENCE.md). Мульти-тулы: [COMPACT_TOOLS.md](COMPACT_TOOLS.md).
