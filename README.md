# Lorne v0.99 — Vi-like Terminal IDE

![](./wiki/image.png)

**Документация:** [wiki/README.md](wiki/README.md) · [wiki/VI_EDITOR.md](wiki/VI_EDITOR.md) · [wiki/KEYBINDINGS.md](wiki/KEYBINDINGS.md) · [wiki/MODES/README.md](wiki/MODES/README.md) · [wiki/PROJECT_BRAIN.md](wiki/PROJECT_BRAIN.md) · [wiki/ARCHITECTURE.md](wiki/ARCHITECTURE.md) · [wiki/TOOLS.md](wiki/TOOLS.md) · [wiki/COMPACT_TOOLS.md](wiki/COMPACT_TOOLS.md) · [wiki/BACKGROUND_AND_DEEP.md](wiki/BACKGROUND_AND_DEEP.md) 

---

## Редактор 

- **Пять режимов:** Normal, Insert, Visual, Command, Widget (управление UI без мыши)
- **Полный набор стандартных vi-клавиш:** hjkl, w/b/e, dd/yy/p, gg/G, :/search, %
- **Widget-режим:** навигация по всем панелям, дереву файлов, чату, вкладкам (`Space+w`)
- **Справка по клавишам:** `F1` или `?` — открывает таблицу всех сочетаний
- **Строка состояния:** режим + подсказки клавиш в правом нижнем углу

## AI-агент

- **Потоковая генерация ответов**, виджет-мысль модели
- **Режимы:** Agent, Ask, Creator, Research, Deep, Brainer — TUI и classic CLI
- **Sub-агенты:** `spawn_subagent`/`get_subagent_result` — фоновый воркер Creator Mode из любого режима (кроме Ask), общий пул (не более 3 одновременно), дерево во вкладке «Активные агенты»
- **9 базовых тулов для слабых моделей** (`structured_memory`, `ast_analyze`, `multi_read`, `lint_check`, `task_decompose`, `env_info`, `batch_replace`, `verify_result`, `session_notes`) плюс **опциональный расширенный набор** мега-тулов (`code_intel_tool`, `workspace_search`, `net_tool`, `viz_tool`, `qa_extended_tool`, `session_meta_tool`, `tools_catalog`, …) — включается тумблером **Extended tools** в настройках, по умолчанию выключен, чтобы не раздувать контекст модели
- **Роли воркеров Creator Mode** с детальными промптами в `Agent/roles/`; воркеры получают укороченный `WORKER_SYSTEM_PROMPT` вместо полного системного промпта сессии

## Интерфейс

- **Волновая анимация** во время работы модели вместо текстовой строки
- **Виджет мыслей модели:** коллапсируемый ThinkingBlock
- **Графики и диаграммы прямо в чате:** ASCII-графики (plotext) и mermaid-flowchart-диаграммы рендерятся из ответа модели
- **Настройки** с компактным интерфейсом
- **TUI (Textual) и classic CLI** — оба режима поддерживаются, `--classic` / `TCA_MODE=classic`

## Project Brain

- **Единая политика по режимам** (`Agent/project_brain/policy.py`): что и когда индексировать/писать/дебаунсить для каждого режима чата
- **Автодокументирование изменений** во всех режимах: Research пишет находки в `agent/research_notes.md`, Creator — сводку прогона в `agent/creator_summary.md`, Deep — итоговый отчёт в `agent/deep_report.md`, Brainer — по ходу диалога
- **Дебаунс полного пересбора** — дорогой AST-скан не гоняется на каждый ход, только когда реально что-то изменилось
- **Relator (опционально) + Python-фолбэк:** `[[TREE]]`, `[[JSON]]`, `%%render%%` — при отсутствии Relator страницы brain всё равно генерируются
- **Автобутстрап** — `project_brain/` создаётся при первом запуске, если его ещё нет
- **Changelog-шаблон:** каждая сессия фиксирует что изменено и когда

---

## Возможности (полный список)

- **Работа с файлами** — чтение , создание, редактирование, поиск по содержимому
- **Терминал** — выполнение shell-команд с подтверждением пользователя
- **Планирование** — автоматическое построение плана для сложных задач с отслеживанием прогресса
- **Версионирование** — SQLite-снимки файлов + Git-интеграция 
- **RAG-поиск** — семантический чанкинг, word-level scoring, mtime-кэш, инкрементальная переиндексация
- **Сессии** — сохранение и восстановление диалогов между запусками, именованные чаты
- **Откат хода (TUI)** — у каждого пользовательского сообщения кнопка отката
- **Много моделей** — 27+ моделей через OpenRouter (бесплатные, дешёвые, платные, про) + Ollama
- **Creator Mode** — параллельное выполнение подзадач несколькими агентами
- **Deep Solver** — длительный автономный режим с чекпоинтами
- **Устойчивость к «петлям»** — подсказки при повторяющихся вызовах одних и тех же инструментов
- **Локальные модели** — доразбор `tool_calls`, восстановление JSON, извлечение `<thought>`
- **Генерация PDF** — создание документов через ReportLab

---

## Быстрый старт

### Требования

- Python 3.10+
- API-ключ [OpenRouter](https://openrouter.ai/)

### Установка

**macOS / Linux:**

```bash
git clone https://github.com/threenebula23/Lorne.git
cd Lorne
chmod +x install.sh
./install.sh
```

**Windows:**

```cmd
git clone https://github.com/threenebula23/Lorne.git
cd Lorne
install.bat
```

Скрипт установки:
1. Создаёт виртуальное окружение `.venv`
2. Устанавливает зависимости из `requirements.txt`
3. Создаёт команды **`lorne`** и **`tca`**  в PATH

### Первый запуск

```bash
# С ключом через аргумент
lorne env=sk-or-v1-ваш_ключ

# Или сохранить ключ в файл
echo 'OPENROUTER_API_KEY=sk-or-v1-ваш_ключ' > Agent/.env
lorne
```

---

## Использование

### Запуск

```bash
lorne                          # работает в текущей директории
lorne /path/to/project         # работает в указанном проекте
lorne env=sk-or-v1-xxx         # передать API-ключ через аргумент
lorne /path/to/project env=KEY # оба варианта (любой порядок)
```

Альтернативный способ запуска (без установки) — тот же разбор `env=` и каталога через `python lorne.py`:

```bash
python lorne.py
python -m Terminal              # TUI; для classic: python -m Terminal --classic
```


### Команды

| Команда | Описание |
|---|---|
| `Enter` (пустой ввод) | Продолжить — агент выполнит следующий шаг |
| `!<команда>` | Выполнить команду в терминале напрямую (например `!ls -la`, `!git status`) |
| `/model` | Выбрать модель из списка (выбор сохраняется) |
| `/model <id>` | Установить произвольную модель по ID |
| `/profile [имя]` | Сменить профиль: `fast`, `balanced`, `quality` |
| `/balance` | Показать баланс OpenRouter |
| `/plan` | Показать текущий план задачи (без LLM) |
| `/status` | Информация о модели, контексте, RAG, сообщениях |
| `/ls [путь]` | Список файлов в директории |
| `/tree [путь]` | Дерево проекта |
| `/rag <запрос>` | Прямой поиск по проекту (RAG) без LLM |
| `/versions <файл>` | История версий файла (SQLite) |
| `/rollback <файл> [id]` | Откатить **один файл** к версии из SQLite (classic и общий сценарий; в TUI дополнительно есть **откат целого хода** — кнопка у сообщения пользователя) |
| `/git status` | Статус Git-репозитория |
| `/git log [файл]` | История Git-коммитов |
| `/git diff [хеш]` | Git diff текущих изменений или коммита |
| `/git rollback <хеш>` | Откатить Git-коммит (revert) |
| `/compact` | Сжать историю разговора (освободить контекст) |
| `/creator` | Включить/выключить Creator Mode (параллельные агенты) |
| `/creator <задача>` | Запустить задачу в Creator Mode |
| `/creator set orchestration …` | `parallel` \| `sequential` \| `supervisor` \| `hierarchical` (см. [EXTENDING.md](wiki/EXTENDING.md)) |
| `/custom` | Управление кастомными инструментами |
| `/agent list` | Список логических под-агентов |
| `/help` | Справка по командам |
| `/exit` | Выйти |

### Сессии

В **TUI** при старте открывается модальное окно со списком чатов: для каждой сессии видны заголовок, время обновления и примерное число сообщений. Доступны **Открыть**, **Удалить**, **Новый чат** и **Выход из TCA**.

В **classic**-режиме по-прежнему текстовый выбор: **Enter** — новая сессия, **номер** — продолжить, **d номер** — удалить.

Сессии и история сообщений хранятся в **`.tca/checkpoints.sqlite`**. Дополнительно для отката ходов ведутся снимки диалога и снимок «версий файлов проекта» на границе каждого пользовательского сообщения (таблицы `turn_snapshots`, `turn_workspace_snapshots`).

---

## Конфигурация

### API-ключ

Три способа указать ключ OpenRouter (в порядке приоритета):

1. **Аргумент запуска:** `lorne env=sk-or-v1-xxx` (или `tca env=…`)
2. **Файл `.env`:** создать `Agent/.env` или `.env` в корне TCA с содержимым `OPENROUTER_API_KEY=sk-or-v1-xxx`
3. **Переменная окружения:** `export OPENROUTER_API_KEY=sk-or-v1-xxx`

### Профили

| Профиль | Temperature | Max tokens | Назначение |
|---|---|---|---|
| `fast` | 0.1 | 4096 | Быстрые простые задачи |
| `balanced` | 0.2 | 8192 | Баланс скорости и качества (по умолчанию) |
| `quality` | 0.1 | 16384 | Максимальное качество |

### Переменные окружения

| Переменная | Описание | По умолчанию |
|---|---|---|
| `OPENROUTER_API_KEY` | API-ключ OpenRouter | — (обязательно) |
| `TCA_MODE` | `tui` (IDE) или `classic` (чат в терминале) | `tui` |
| `TCA_PROFILE` | Профиль по умолчанию | `balanced` |
| `TCA_MODEL` | Модель по умолчанию | `arcee-ai/trinity-large-preview:free` |
| `TCA_BASE_URL` | Base URL для API | `https://openrouter.ai/api/v1` |
| `TCA_MODEL_FAST` | Модель для профиля fast | значение `TCA_MODEL` |
| `TCA_MODEL_BALANCED` | Модель для профиля balanced | значение `TCA_MODEL` |
| `TCA_MODEL_QUALITY` | Модель для профиля quality | значение `TCA_MODEL` |
| `TCA_TEMP_FAST` | Temperature для fast | `0.1` |
| `TCA_TEMP_BALANCED` | Temperature для balanced | `0.2` |
| `TCA_TEMP_QUALITY` | Temperature для quality | `0.1` |
| `TCA_MAX_TOKENS` | Max tokens (глобально) | `8192` |
| `TCA_MAX_TOKENS_FAST` | Max tokens для fast | `4096` |
| `TCA_MAX_TOKENS_BALANCED` | Max tokens для balanced | `8192` |
| `TCA_MAX_TOKENS_QUALITY` | Max tokens для quality | `16384` |
| `TCA_RAG_PATTERNS` | Паттерны для RAG-индексации | `*.py,*.md,*.ts,*.tsx,*.json` |
| `TCA_RAG_MAX_FILES` | Макс. число файлов для RAG | `500` |
| `TCA_RUN_COMMAND_DEDUPE_S` | Окно анти-спама для **повторной той же** `run_command` (сек.); **0** = отключено | `0` |
| `LOCAL_API_KEY` | API-ключ для локального сервера (Creator Mode) | — |

### Доступные модели

Команда `/model` показывает полный список. Краткая сводка:

**Бесплатные:**
Trinity Large, Step 3.5 Flash, Qwen3 235B Thinking

**Платные:**
Qwen3 235B Thinking, Qwen3 Coder 30B, Qwen3.5 Flash, GPT OSS 120B, GPT-5 Nano, Gemini 2.5 Flash Lite

**Доступные (cheap):**
Qwen3 Coder Next, Qwen3.5 35B, Qwen3 Coder, Qwen3.5 Plus, Qwen3.5 397B, GPT-4o Mini, GPT-5 Mini, Gemini 2.5 Flash, Gemini 3 Flash, Grok 4.1 Fast, Grok Code Fast, DeepSeek V3.2

**Про (pro):**
GPT-5.1 Codex, GPT-5.3 Codex, Gemini 3.1 Pro, Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.6

Выбор модели сохраняется в `~/.tca_config.json` между запусками.

---

## Архитектура

Подробная карта модулей, потоков данных и путей к SQLite — в **[wiki/ARCHITECTURE.md](wiki/ARCHITECTURE.md)**.

### Структура проекта (кратко)

```
TCA/
├── tca.py                      # Точка входа Python; после install — команды lorne и tca (алиас)
├── requirements.txt
├── wiki/                       # ARCHITECTURE.md, EXTENDING.md, TOOLS.md, BACKGROUND_AND_DEEP.md
│
├── Agent/                      # Ядро: LLM, LangGraph, инструменты, RAG, сессии
│   ├── agent/                  # run_tui_mode / run_coding_agent_loop; снимки, откат TUI
│   ├── graph_runner.py         # LangGraph: call_model, execute_tools, анти-петля
│   ├── tool_registry.py        # build_tools(agent_mode, playwright_python), compact + custom
│   ├── message_utils/          # Санитизация, компактирование, восстановление tool JSON, петли тулов
│   ├── deep_solver/            # Пакет Deep Solver; legacy_loop.py — долгий цикл, чекпоинты
│   ├── background_agent_runner.py  # Фоновый LLM+тул-цикл для start_background_task
│   ├── command_router/         # Slash-команды в classic-режиме
│   ├── llm_provider.py         # OpenRouter, профили, модели
│   ├── planner.py              # Планы задач
│   ├── git_integration.py      # GitPython
│   ├── creator_mode.py         # Creator: воркеры, оркестрация, супервайзер
│   ├── creator_orchestration.py  # Роли, handoff, сводка супервайзера
│   ├── creator_summary.py      # Единый текст итога Creator (TUI + classic + сессия)
│   ├── creator_provider.py     # Конфиг Creator (orchestration, local/heavy)
│   ├── system_prompt.py        # SYSTEM_PROMPT (сессия) + WORKER_SYSTEM_PROMPT (Creator-воркеры)
│   ├── prompts/                # Промпты режимов (*.md) + project_brain_rules.py
│   ├── subagent_runner.py      # Общий раннер фоновых суб-агентов (spawn/get_result, лимит 3)
│   ├── tools/                  # @tool + compact_tools.py, extended_tools.py (опциональный тир)
│   ├── rag/                    # Индексация и rag_search
│   ├── project_brain/          # Скан, Markdown brain, policy.py — политика по режимам
│   ├── checkpoint/             # Сессии (SQLite)
│   ├── versioning/             # Снимки файлов (SQLite)
│   └── file_loading/           # Загрузка файлов для RAG
│
├── Interface/                  # TUI (Textual) + Rich для classic
│   ├── tui_app.py              # LorneApp: layout IDE
│   ├── session_picker_screen.py  # Модальный выбор сессии при старте TUI
│   ├── tui_bridge.py           # Мост агент ↔ панели (потокобезопасно)
│   ├── themes.py               # Темы
│   ├── visualization.py        # Rich в classic-режиме
│   ├── graph_display.py        # Creator Mode (classic)
│   └── panels/                 # file_explorer, active_agents, workspace_center (чат + вкладки), code_editor, …
│
└── Terminal/                   # python -m Terminal — те же режимы, что у tca.py
    ├── cli.py
    └── runner.py
```

### Как работает агент

TCA построен на [LangGraph](https://github.com/langchain-ai/langgraph) — фреймворке для создания графов состояний поверх LangChain.

#### Граф выполнения

```
┌────────────┐     tool_calls?   ┌─────────┐
│    agent   │ ──── yes ────────▶│  tools  │
│(call_model)│ ◀─────────────────│(execute)│
└────────────┘                   └─────────┘
       │
       │ no tool_calls
       ▼
      END
```

1. **`call_model`** — отправляет историю сообщений в LLM, получает ответ. Перед отправкой вызывает `_sanitize_messages()` для исправления возможных повреждений истории. При ошибке провайдера автоматически повторяет запрос (до 2 раз).

2. **`should_continue`** — если в ответе есть `tool_calls`, переходит к узлу `tools`. Иначе — завершение (END).

3. **`execute_tools`** — выполняет вызванные инструменты (read-only параллельно в пуле потоков, если все вызовы из «безопасного» набора; иначе по очереди), формирует `ToolMessage` с результатами. Если провайдер не поддерживает `parallel_tool_calls` при `bind_tools`, выполняется повторная привязка без этого флага.

4. Цикл повторяется, пока модель не ответит текстом без tool_calls.

#### Поток данных одного хода

```
Пользователь вводит задачу
  │
  ▼
Планирование (build_plan) → plan_tool(action=save) → .tca/plan.json
  │
  ▼
HumanMessage добавляется в messages
  │
  ▼
app.stream(messages) запускает граф:
  │
  ├─ call_model → AIMessage(tool_calls=[edit_file, run_command])
  ├─ execute_tools → [ToolMessage(result1), ToolMessage(result2)]
  ├─ call_model → AIMessage(tool_calls=[plan_tool update])
  ├─ execute_tools → [ToolMessage(result)]
  ├─ call_model → AIMessage(content="Готово! Вот что я сделал...")
  └─ END
  │
  ▼
save_state(messages) → SQLite
```

### Система инструментов

Инструменты — `@tool` (LangChain). У модели — **компактные имена** (`plan_tool`, `git_ops`, `library_context`, …) плюс «атомарные» (`read_file`, `edit_file`, `web_search`, …). Подробная таблица и поля `action`: **[wiki/TOOLS.md](wiki/TOOLS.md)** и **[wiki/COMPACT_TOOLS.md](wiki/COMPACT_TOOLS.md)**.

Кратко:

| Группа | Примеры имён у модели |
|--------|------------------------|
| Файлы | `read_file`, `list_files`, `edit_file`, `write_file`, `replace_file_lines`, `insert_file_lines`, `search_in_files`, **`code_file_tool`** |
| План / мысли | **`plan_tool`**, **`reasoning_tool`** |
| Терминал / код | `run_command`, `code_interpreter` |
| Git / версии | **`git_ops`**, **`file_versions_tool`** |
| Веб / доки | `web_search`, `web_fetch`, **`library_context`** (в т.ч. `action=search` вместо отдельного get_documentation) |
| Office / OCR | `office_document_read`, **`docx_write_tool`**, `docx_document_advanced_ops`, **`docxedit_tool`**, **`ocr_tool`**, `pdf_styled_document_create`, `create_pdf` |
| Прочее | `rag_search`, `ask_user`, кастомные тулы |
| **Только TUI + режим Agent** | **`headless_browser`**, **`playwright_sync`** (Python — только при галочке в **Files → Settings**) |
| **Фон** | **`start_background_task`**, **`get_background_result`** — см. [BACKGROUND_AND_DEEP.md](wiki/BACKGROUND_AND_DEEP.md) |
| **Суб-агенты** | **`spawn_subagent`**, **`get_subagent_result`** — фоновый Creator-воркер из Agent/Brainer/Research/Deep; недоступны в Ask |
| **Загрузки** | **`download_file`** — HTTP(S) в файл в рабочей области проекта, лимиты в схеме (`Agent/tools/download_tool.py`) |
| **Расширенный тир (опционально)** | `code_intel_tool`, `workspace_search`, `net_tool` (http/port_check/db_query), `viz_tool` (chart/diagram), `qa_extended_tool`, `session_meta_tool`, `tools_catalog`, `diff_tool`, `apply_patch`, `project_tree`, `brain_search`, `export_to_brain`, `memory_search` — включается тумблером **Extended tools**, по умолчанию выключен ради бюджета контекста |

Защита `run_command`: блокировка опасных команд; **опциональная** дедупликация повторов одной и той же команды — только если задано ненулевое `TCA_RUN_COMMAND_DEDUPE_S` (по умолчанию выключено). Снимки версий — перед правками файлов.

Детали фонового помощника, **Deep Solver** и суб-агентов (`spawn_subagent` / `get_subagent_result`): **[wiki/BACKGROUND_AND_DEEP.md](wiki/BACKGROUND_AND_DEEP.md)**.

### Управление контекстом

LLM имеют ограниченное окно контекста. TCA управляет этим через:

- **Компактирование** (`compact_conversation`) — старые сообщения сжимаются в текстовое резюме, сохраняя последние 10–12 сообщений. При сжатии границы не разрывают группы tool_call/ToolMessage.
- **Авто-компактирование** — срабатывает автоматически при превышении 30 сообщений **или** ~85% лимита контекста модели (TUI и classic).
- **Тиринг тулов** — базовый набор всегда в промпте; расширенный тир (мега-тулы) — опционально по тумблеру **Extended tools**, чтобы JSON-схемы не раздували системный промпт.
- **Слим-промпты по режимам** — общий `SYSTEM_PROMPT` минимален, детали конкретного режима (Agent/Ask/Brainer/…) добавляются отдельным фрагментом (`Agent/prompts/*.md`); воркеры Creator получают отдельный укороченный `WORKER_SYSTEM_PROMPT`.
- **Усечение результатов** (`_truncate_result`) — большие ответы инструментов обрезаются (лимиты по инструментам: 2000–4000 символов).
- **Санитизация** (`_sanitize_messages`) — перед каждым вызовом LLM проверяет и исправляет историю: удаляет осиротевшие `ToolMessage`, добавляет заглушки для незавершённых `tool_calls`.

### Устойчивость к ошибкам

- **Ретраи провайдера** — при ошибках вроде «Provider returned error», «rate limit», «bad gateway» автоматический повтор с задержкой (до 2 попыток). OpenRouter перемаршрутизирует на другого провайдера.
- **Восстановление JSON** — сломанный JSON от маленьких моделей восстанавливается через `json-repair` и ручной парсинг.
- **Починка tool_calls** — если модель возвращает tool_calls как текст (JSON в content), TCA парсит их вручную.
- **Склейка контента** — если модель ломает многострочный код в JSON-аргументах, `_reconstruct_broken_content` собирает фрагменты обратно.
- **Нормализация кода** — если модель передаёт литералы `\n` вместо переносов строк, они конвертируются в реальные переводы.
- **Петли тулов** — при многократно одинаковых вызовах в историю может вставляться нudge «смени стратегию» (`web_search`, план, другой файл/команда), см. `message_utils.tool_repetition_loop_nudge`.

### Хранение данных

| Файл | Формат | Содержимое |
|---|---|---|
| `.tca/checkpoints.sqlite` | SQLite | Сессии: `sessions`, `checkpoints` (messages JSON), снимки для отката ходов (`turn_*`) |
| `.tca/versions.sqlite` | SQLite | Снимки файлов для отката |
| `.tca/plan.json` | JSON | Текущий план задачи (`planning_tool`) |
| `.tca/ui_settings.json` | JSON | Тема, плотность, подсветка; браузерные тулы; свои модели OpenRouter/Ollama; пресеты Ollama |
| `~/.tca_config.json` | JSON | Выбранная модель, настройки Creator и др. |

Файлы под `.tca/` создаются в рабочей директории проекта; глобальный конфиг — в домашнем каталоге.

---

## Разработка

### Установка для разработки

```bash
git clone https://github.com/your-repo/TCA.git
cd TCA
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-v1-ваш_ключ' > Agent/.env
python tca.py
```

### Зависимости

```
python-dotenv    — загрузка .env файлов
json-repair      — восстановление сломанного JSON от LLM
langchain-core   — базовые абстракции (messages, tools)
langchain-openai — ChatOpenAI для работы с OpenRouter
langgraph        — граф состояний для agent loop
rich             — красивый терминальный вывод
gitpython        — интеграция с Git (авто-коммиты, rollback, история)
ddgs             — веб-поиск через DuckDuckGo
reportlab        — генерация PDF (опционально)
playwright       — Python API браузера в режиме Agent (после установки: `playwright install chromium`)
```

### Добавление нового инструмента

1. Создайте файл в `Agent/tools/`, например `my_tool.py`:

```python
from langchain_core.tools import tool

@tool
def my_tool(arg1: str, arg2: int = 10) -> dict:
    """Описание инструмента — агент увидит этот текст."""
    # Логика
    return {"ok": True, "result": "..."}
```

2. Экспортируйте из `Agent/tools/__init__.py`:

```python
from .my_tool import my_tool
# Добавьте в __all__
```

3. Добавьте в список `_base_tools` в `Agent/tool_registry.py`:

```python
_base_tools: List[Any] = [
    ...,
    my_tool,
]
```

4. Опционально: добавьте описание в системный промпт (`Agent/system_prompt.py`) или в промпт конкретного режима (`Agent/prompts/*.md`), и специальный вывод в `Interface/visualization.py`.

### Добавление модели

Добавьте запись в `AVAILABLE_MODELS` в `Agent/llm_provider.py`:

```python
{"id": "provider/model-name", "name": "Display Name", "ctx": 128_000, "tier": "free|cheap|paid|pro"},
```

Если провайдер модели поддерживает `parallel_tool_calls`, добавьте его в `_PROVIDER_CAPS`.

### Ключевые модули для разработчика

| Модуль | Что менять |
|---|---|
| `Agent/agent/` | Точка входа TUI/classic, мост с UI, сессии |
| `Agent/graph_runner.py` | LangGraph: узлы и рёбра графа, подсказки при петлях |
| `Agent/tool_registry.py` | Список инструментов, `build_tools(agent_mode=...)` |
| `Agent/command_router/` | Slash-команды (classic) |
| `Agent/deep_solver/` | Режим Deep Solver (локальная модель), `legacy_loop.py` |
| `Agent/background_agent_runner.py` | Очередь фоновых микро-задач LLM+тулов |
| `Agent/message_utils/` | Санитизация, компактирование, восстановление tool JSON, анти-петля |
| `Agent/git_integration.py` | Git |
| `Agent/rag/` | RAG |
| `Agent/llm_provider.py` | Модели и OpenRouter |
| `Agent/system_prompt.py` | `SYSTEM_PROMPT` / `WORKER_SYSTEM_PROMPT` |
| `Agent/prompts/` | Промпты режимов (`*.md`), `project_brain_rules.py` |
| `Agent/project_brain/` | Скан, Markdown brain, `policy.py` (политика по режимам) |
| `Agent/subagent_runner.py` | Фоновые суб-агенты: `spawn`/`get_result`, лимит concurrency |
| `Agent/tools/` | Реализации инструментов (+ `extended_tools.py` — опциональный тир) |
| `Interface/tui_app.py` | Layout IDE, CSS |
| `Interface/tui_bridge.py` | Обновление панелей из фонового агента |
| `Interface/panels/*.py` | Панели: дерево, агенты, чат, редактор, … |
| `Interface/visualization.py` | Вывод в classic-режиме |
| `Terminal/runner.py` | Shell-команды |

Полная таблица файлов — **[wiki/ARCHITECTURE.md](wiki/ARCHITECTURE.md)**.

---

## Удаление

**macOS / Linux:**

```bash
./uninstall.sh
```

**Windows:**

```cmd
uninstall.bat
```

Скрипт удалит виртуальное окружение и команды `lorne` / `tca` в PATH. Опционально удалит данные сессий и версий.

---

## Лицензия

MIT — см. [LICENSE](LICENSE).
