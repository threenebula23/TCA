### Режим Agent

Полный цикл с тулами. Дисциплина: перед сложной задачей — `task_decompose`;
`list_files`/`search_in_files` → `read_file` / `multi_read` → `ast_analyze` для структуры →
`rag_search` для архитектуры → `plan_tool` на многошаговые задачи → правки
(`replace_file_lines`/`batch_replace`/`write_file`/`code_file_tool`) →
`lint_check` после правок → `verify_result` перед отчётом;
факты → `structured_memory`.

Строковые JSON-аргументы (`steps_json`, `data_json`, …): одна строка = **полный** валидный JSON в **двойных** кавычках, без обрыва; для `plan_tool` save надёжнее поле `steps` как массив строк.

**Новые инструменты (v1.0):** `task_decompose` — декомпозиция задачи;
`structured_memory` — KV-память сессии; `multi_read` — читать 8 файлов за раз;
`ast_analyze` — структура кода без чтения файла; `lint_check` — линтер после правок;
`verify_result` — проверка условий задачи; `batch_replace` — пакетная замена;
`env_info` — окружение; `session_notes` — свободные заметки.
