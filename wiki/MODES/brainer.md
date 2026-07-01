# Режим: brainer

## Реализация

- Промпт-дополнение: `Agent/prompts/brainer.md` — workflow (`rag_search` → `read_file` по `project_brain/*.md` → `ast_analyze`/`multi_read` по коду → правки → `write_brain`), критерии когда писать в brain, и требование короткой записи в `agent/session_notes.md`, если за ход не было ни одного `write_brain`.
- Политика brain (`Agent/project_brain/policy.py`, `"brainer"`): `rag_search`/`project_brain_tool` форсируются в набор тулов (`forced_tool_names`) даже при выключенных custom tools; переиндексация RAG — после каждого раунда с тулами; полный `refresh` — после хода, но **с дебаунсом** (см. ниже).

## Схема потока

```mermaid
flowchart LR
  userNode[User] --> ragNode[RAG brain first]
  ragNode --> readNode[read_file project_brain]
  readNode --> codeNode[Source files]
```

## Инструменты

`project_brain_tool` (refresh / write_brain / write_architecture / …), `rag_search`, чтение файлов — в приоритете. См. [PROJECT_BRAIN.md](../PROJECT_BRAIN.md).

## Автообновление brain

В коде графа (`AgentGraph._brain_sync`): после каждого раунда с тулами выполняется переиндексация RAG с диска; после финального ответа без тулов — полный `refresh_project_brain` (скан) + переиндексация, **но только если** `should_full_refresh()` считает его нужным (изменился changelog или прошло достаточно времени с прошлого полного скана — `Agent/project_brain/policy.py`, дебаунс по умолчанию 120с, `LORNE_BRAIN_REFRESH_DEBOUNCE_SEC`/`TCA_BRAIN_REFRESH_DEBOUNCE_SEC`). После полного refresh краткая выжимка brain, вшитая в системный промпт сессии, обновляется на лету (`SystemMessage(id="lorne_brain_excerpt")`). Ошибки синхронизации больше не проглатываются молча — попадают в UI как предупреждение. При остановке пользователем (TUI / classic) — тот же полный refresh (с учётом дебаунса), если режим Brainer.

`project_brain/` создаётся автоматически при первом запуске сессии (`bootstrap_project_brain`), если каталога ещё нет — не нужно вручную вызывать `refresh` перед первым использованием.
