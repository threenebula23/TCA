# Module: Agent.checkpoint

## Purpose

Python module `Agent.checkpoint`.

---

## Responsibilities

- Python module `Agent.checkpoint`.

---

## Public API

|name|description|
|---|---|
|save_state|Сохраняет список сообщений (LangChain или dict) в SQLite.|
|load_state|Загружает состояние из SQLite. Возвращает список dict (type, content, ...) или None.|
|create_session|Создаёт новую сессию и возвращает session_id.|
|list_sessions|Список сохранённых чатов (последние сверху).|
|delete_session|Удаляет чат (sessions + checkpoints).|
|save_pre_turn_snapshot|Состояние диалога до добавления turn_index-го пользовательского сообщения (0 = до первого Human).|
|load_pre_turn_snapshot||
|delete_turn_snapshots_from|Удаляет снимки с turn_index >= from_turn_index (после отката).|
|get_session_created_at|ISO created_at строки sessions (начало «жизни» чата для скоупа отката файлов).|
|workspace_mapping_for_turn|path→version_id: только под корнем проекта; turn_index>0 — только файлы с версией TCA не раньше created_at сессии.|
|save_pre_turn_workspace_snapshot|Снимок path→version_id для отката: не глобально по всей БД, а по текущему проекту и (с 2-го хода) по файлам сессии.|
|delete_turn_workspace_snapshots_from||
|restore_turn_workspace|Восстанавливает файлы по снимку turn_index: откат версий + удаление файлов, созданных после метки.|
|messages_from_stored_dicts|Восстанавливает LangChain-сообщения из JSON checkpoint + свежий system prompt.|

---

## Dependencies

- `Agent.message_utils`
- `Agent.path_utils`
- `Agent.runtime_paths`
- `Agent.versioning`
- `datetime`
- `json`
- `langchain_core.messages`
- `message_utils`
- `path_utils`
- `pathlib`
- `sqlite3`
- `typing`
- `uuid`
- `versioning`

---

## Used By

- `Agent/agent/_impl_prepare.py`
- `Agent/deep_solver/_impl_a.py`
- `Agent/deep_solver/_impl_b.py`
- `Agent/deep_solver/legacy_loop.py`
- `Interface/tui_app.py`
- `tests/test_checkpoint_rollback.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/checkpoint/__init__.py`

---

## Entry Points


---

## API / route hints

