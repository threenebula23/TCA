### Режим Research

Опора на внешние источники: `web_search` → `web_fetch` для деталей; `library_context`
для версий API пакетов; при связи с кодом репозитория — `rag_search` и `multi_read`;
фиксируй находки в `session_notes` с тегом 'research'; `env_info` для проверки окружения.

Строковые JSON-аргументы (`steps_json`, `data_json`, …): одна строка = **полный** валидный JSON в **двойных** кавычках, без обрыва; для `plan_tool` save надёжнее поле `steps` как массив строк.

**Новые инструменты (v1.0):** `task_decompose` — декомпозиция задачи;
`structured_memory` — KV-память сессии; `multi_read` — читать 8 файлов за раз;
`ast_analyze` — структура кода без чтения файла; `lint_check` — линтер после правок;
`verify_result` — проверка условий задачи; `batch_replace` — пакетная замена;
`env_info` — окружение; `session_notes` — свободные заметки.
