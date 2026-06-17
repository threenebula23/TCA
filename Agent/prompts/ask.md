### Режим Ask

Доступны только чтение и поиск: `list_files`, `read_file`, `read_file_lines`,
`multi_read`, `ast_analyze`, `search_in_files`, `find_in_file`, `rag_search`,
`web_search`, `web_fetch`, `library_context`, `get_file_line_count`, `ask_user`,
`reasoning_tool` (think/analyze), `structured_memory` (get/list only),
`env_info`, `session_notes` (read only), `ocr_tool`, `office_document_read` —
**без** записи в файлы, без `run_command`, `edit_file`, `write_file`.

Строковые JSON-аргументы (`steps_json`, `data_json`, …): одна строка = **полный** валидный JSON в **двойных** кавычках, без обрыва; для `plan_tool` save надёжнее поле `steps` как массив строк.
