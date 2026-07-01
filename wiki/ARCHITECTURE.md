# Архитектура TCA (Lorne)

Краткая карта для разработчиков. Детали режимов: [MODES/README.md](MODES/README.md). Brain: [PROJECT_BRAIN.md](PROJECT_BRAIN.md).

## Запуск

| Режим | Env | Вход | UI |
|--------|-----|------|-----|
| TUI | `LORNE_MODE` / `TCA_MODE` по умолчанию `tui` | `lorne.py` → `Agent.agent.run_tui_mode()` (`_impl_tui.py`) | Textual: `Interface/tui_app.py` |
| Classic CLI | `--classic` / `LORNE_MODE=classic` \| `TCA_MODE=classic` | `Agent.agent.run_coding_agent_loop()` (`_impl_classic.py`) | Rich: `Interface/visualization.py` |

Оба режима идут через один и тот же `Agent/graph_runner.py` + `Agent/tool_registry.py`; общая логика (bootstrap Project Brain, форсирование brain-тулов в Brainer, авто-компактирование истории) вынесена в `Agent/agent/_impl_prepare.py` и вызывается из обоих `_impl_tui.py`/`_impl_classic.py`.

`python -m Terminal` и `tca.py` делегируют в `lorne.main`. См. [tutorials/quickstart.md](tutorials/quickstart.md).

## Поток данных (TUI)

```
lorne.py → Agent.agent (TUI) → Interface.LorneApp + TUIBridge
         → LangGraph (Agent/graph_runner.py) → tool_registry.build_tools → tools
```

Откат хода: `Agent/checkpoint/` + `Agent/versioning/`; SQLite в `project_data_dir`: `checkpoints.sqlite`, `versions.sqlite`. Пути: `Agent/runtime_paths.py` (`.lorne` приоритетно, иначе `.tca`).

## Основные пакеты

| Пакет | Роль |
|-------|------|
| `Agent/graph_runner.py` | Узлы графа, `execute_tools`, read-only параллельно |
| `Agent/tool_registry.py` | `_base_tools`, `build_tools`, Ask/Agent флаги |
| `Agent/tool_schemas.py` | Pydantic аргументы, coerce |
| `Agent/tools/` | Реализации `@tool`, `compact_tools.py` — мульти-тулы, `extended_tools.py` — опциональный тир |
| `Agent/rag/` | Индексация, `rag_search` |
| `Agent/project_brain/` | Скан, Markdown brain, `write_brain_markdown`, `policy.py` (политика по режимам чата) |
| `Agent/subagent_runner.py` | Общий фоновый раннер суб-агентов (`spawn`/`get_result`, лимит 3 одновременно) |
| `Agent/deep_solver/` | Deep Solver (отдельный цикл, локальная модель); суб-агенты — через `subagent_runner` |
| `Agent/creator_*.py` | Creator Mode; воркеры используют `WORKER_SYSTEM_PROMPT` |
| `Agent/prompts/` | Промпты режимов (`*.md`), `project_brain_rules.py` |
| `Interface/` | Textual UI, `tui_bridge.py`, `panels/` (в т.ч. рендер графиков/диаграмм из ответа модели) |
| `Terminal/` | CLI-обёртка над `lorne` |

## Панели TUI

| Модуль | Назначение |
|--------|------------|
| `Interface/panels/ai_chat/` | Чат, режим, ввод, откат |
| `Interface/panels/workspace_center.py` | Вкладки чат + редактор |
| `Interface/panels/file_explorer.py` | Дерево, настройки |
| `Interface/panels/active_agents_panel.py` | Дерево агентов: Creator-воркеры и фоновые суб-агенты (`spawn_subagent`) из любого режима |
| `Interface/panels/rich_message.py` | Парсинг fenced-блоков ` ```lorne-chart `/` ```mermaid ` из ответа модели |
| `Interface/panels/blocks/chart_block.py` | ASCII-график (plotext) из `lorne-chart` JSON |
| `Interface/panels/blocks/diagram_block.py` | ASCII flowchart из mermaid-lite текста |
| `Interface/panels/vi_textarea.py` | Vi-like редактор (5 режимов) |
| `Interface/panels/thinking_block.py` | Виджет мыслей модели |
| `Interface/panels/keybindings_data.py` | Данные всех клавиш (одна точка истины) |

Стили: `Interface/tui_app.tcss`, темы: `Interface/themes.py`.

## Тулы для слабых моделей (базовый набор)

| Инструмент | Назначение |
|------------|------------|
| `structured_memory` | Сессионная KV-память |
| `ast_analyze` | Структура кода без чтения файла |
| `multi_read` | Чтение 8 файлов за 1 вызов |
| `lint_check` | Линтер (ruff/eslint/tsc) |
| `task_decompose` | Декомпозиция сложной задачи |
| `env_info` | Окружение: Python, пакеты, команды |
| `batch_replace` | Пакетная замена в файле |
| `verify_result` | Проверка результата (синтаксис, строки) |
| `session_notes` | Свободные заметки сессии; `tag=research` экспортируется в brain в режиме Research |

## Опциональный расширенный тир (`Agent/tools/extended_tools.py`)

Мега-тулы с полем `action`, объединяющие несколько операций под одним именем/схемой — снижают размер JSON-схем, отправляемых модели. Включаются тумблером **Extended tools** (`extended_tools_enabled` в prefs, по умолчанию `false`); брейнер-тулы (`rag_search`, `project_brain_tool`) при этом всё равно форсируются в режиме Brainer через `Agent/project_brain/policy.py`.

| Инструмент | Действия (`action`) |
|------------|----------------------|
| `code_intel_tool` | AST/структура кода |
| `workspace_search` | Поиск по репозиторию |
| `net_tool` | `http` \| `port_check` \| `db_query` (SQLite, только `SELECT`) |
| `viz_tool` | `chart` (данные для `lorne-chart`) \| `diagram` (mermaid-lite) |
| `qa_extended_tool` | `test` \| `lint` |
| `session_meta_tool` | `transcript_search` \| `config_inspect` \| `json_validate` \| `review_checklist` \| `notify_when_done` |
| `tools_catalog` | `list` \| `describe` — справочник по всем тулам по требованию, без раздувания промпта |
| `diff_tool`, `apply_patch`, `project_tree`, `brain_search`, `export_to_brain`, `memory_search` | см. `Agent/tools/extended_tools.py` |

## Суб-агенты (`Agent/subagent_runner.py`)

`spawn_subagent(task)` запускает фоновый Creator-воркер (общий пул, не более 3 одновременно); `get_subagent_result(token, wait_seconds)` — опрос/ожидание результата. Доступны в Agent/Brainer/Research/Deep — **исключены в Ask** (`_ASK_EXCLUDED_TOOL_NAMES`). Дерево `main → children` отображается в `ActiveAgentsPanel` через `TUIBridge.on_subagent_spawn`/`on_subagent_done`; Deep Solver использует тот же раннер вместо собственной копии.

## Данные на диске

| Путь | Содержимое |
|------|------------|
| `project_data_dir` / `ui_settings.json` | prefs UI |
| `project_data_dir` / `*.sqlite` | checkpoints, versions |
| `~/.lorne_config.json` | глобальные настройки (legacy: `~/.tca_config.json`) |
| `Agent/.env` | ключи API (dotenv) |

## Дальше

- [TOOLS.md](TOOLS.md), [COMPACT_TOOLS.md](COMPACT_TOOLS.md), [tool/REFERENCE.md](tool/REFERENCE.md)
- [Interface/OVERVIEW.md](Interface/OVERVIEW.md), [Interface/SETTINGS.md](Interface/SETTINGS.md)
- [developer/ADDING_TOOLS.md](developer/ADDING_TOOLS.md)
