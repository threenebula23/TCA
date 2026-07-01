MODULE: Agent.versioning

PURPOSE:
Версионирование файлов для отката правок (SQLite).

PUBLIC_API:
|name|description|
|---|---|
|save_version|Сохраняет версию содержимого файла. Возвращает version_id.|
|list_versions|Список версий для файла (последние сначала).|
|get_version_content||
|rollback_to_version|Откат файла к конкретной версии.|
|rollback_last|Откат к самой последней сохранённой версии.|
|latest_version_id_for_path|ID последней версии по пути (по created_at).|
|latest_version_created_at|created_at последней версии по пути (ISO), или None.|
|snapshot_all_paths_latest_version|Для каждого пути в БД версий — id самой новой версии (снимок «текущего» состояния трека TCA).|
|paths_first_version_strictly_after|Пути, у которых первая запись в file_versions новее ts_iso (файл впервые отслежен после метки времени).|

DEPENDENCIES:
- Agent.runtime_paths
- __future__
- datetime
- hashlib
- pathlib
- sqlite3
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/checkpoint/__init__.py
- Agent/tools/versioning_tool.py
- Interface/panels/diff_block.py
- tests/test_checkpoint_rollback.py
- tests/test_ollama_provider.py

RISKS:
