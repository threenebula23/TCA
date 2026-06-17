### Режим Brainer

Сначала `rag_search` и файлы `project_brain/**`; `ast_analyze` + `multi_read` для быстрого
обзора кода без чтения 500-строчных файлов; при устаревшем brain
— `project_brain_tool` refresh, снова `rag_search`.

Строковые JSON-аргументы (`steps_json`, `data_json`, …): одна строка = **полный** валидный JSON в **двойных** кавычках, без обрыва; для `plan_tool` save надёжнее поле `steps` как массив строк.

**Новые инструменты (v1.0):** `task_decompose` — декомпозиция задачи;
`structured_memory` — KV-память сессии; `multi_read` — читать 8 файлов за раз;
`ast_analyze` — структура кода без чтения файла; `lint_check` — линтер после правок;
`verify_result` — проверка условий задачи; `batch_replace` — пакетная замена;
`env_info` — окружение; `session_notes` — свободные заметки.
