### Режим Agent

Полный цикл с тулами. `list_files`/`search_in_files` → `read_file`/`multi_read` →
`ast_analyze` для структуры → `rag_search` для архитектуры → `plan_tool` на
многошаговые задачи → правки (`replace_file_lines`/`batch_replace`/`write_file`) →
`lint_check` после правок → `verify_result` перед отчётом; факты → `structured_memory`.
