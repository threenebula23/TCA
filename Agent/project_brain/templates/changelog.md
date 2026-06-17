# Changelog — [[project_name]]

Автоматически генерируется Lorne на основе `.lorne/brain_changelog.jsonl`.
Каждая запись соответствует одному вызову инструмента, изменившего файл.

---

## Последние изменения

[[LIST changelog_entries AS entry]]
- **[[entry.timestamp]]** `[[entry.mode]]` — `[[entry.tool_name]]` → `[[entry.file_path]]`
  [[entry.action]][[/LIST]]

---

*Всего записей: [[len:changelog_entries]]*
