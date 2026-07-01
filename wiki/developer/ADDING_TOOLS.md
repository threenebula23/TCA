# Добавление и обновление инструмента (tool)

Чеклист по репозиторию. Обновление существующего тула — те же шаги, что относятся к изменению.

## 1. Реализация

- Новый модуль или функция в **`Agent/tools/`** с декоратором **`@tool`** (LangChain).
- Экспорт в **`Agent/tools/__init__.py`**, если принято для импорта из пакета.

## 2. Реестр

- **`Agent/tool_registry.py`**: добавить объект в **`_base_tools`** (всегда доступен) или в **`_extended_tools`** (опциональный тир, ` Agent/tools/extended_tools.py`, включается тумблером Extended tools — используйте для тулов, которые не нужны в каждой сессии, чтобы не раздувать бюджет схем).
- Для Ask-режима: при необходимости добавить имя в **`_ASK_EXCLUDED_TOOL_NAMES`** или оставить доступным.
- Для переключателя Custom tools: при необходимости **`_CUSTOM_TOOL_NAMES`**.
- Если тул обязателен в конкретном режиме независимо от prefs (как `rag_search`/`project_brain_tool` в Brainer) — добавить в `force_tools` соответствующей записи `BRAIN_POLICY` (`Agent/project_brain/policy.py`) и убедиться, что `_ensure_forced_brain_tools` подхватывает его.

## 3. Схемы и coerce

- **`Agent/tool_schemas.py`**: класс `*Args(BaseModel)`, зарегистрировать в **`TOOL_ARG_MODELS`**.
- При типичных ошибках модели с аргументами — расширить **`_coerce_common_arg_mistakes`**.

## 4. Компактный диспетчер

- Если тул объединяется с другими под одним именем — ветка в **`Agent/tools/compact_tools.py`**.
- Обновить **[COMPACT_TOOLS.md](../COMPACT_TOOLS.md)** и секцию в **[tool/REFERENCE.md](../tool/REFERENCE.md)**.

## 5. Поведение модели

- При необходимости: **`Agent/system_prompt.py`** (общий `SYSTEM_PROMPT`/`WORKER_SYSTEM_PROMPT`) или **`Agent/prompts/<mode>.md`** (дополнение конкретного режима). Держите оба короткими — есть тесты бюджета (`tests/test_prompt_budget.py`), которые проверяют размер промптов и схем.

## 6. UI

- Карточка результата: **`Interface/panels/tool_card.py`** (ветка по `tool_name`).

## 7. Тесты

- Добавить smoke в **`tests/`** по аналогии с `test_project_brain_tool.py` или существующими тул-тестами.

## 8. Документация

- Секция в **[wiki/tool/REFERENCE.md](../tool/REFERENCE.md)** (или отдельная страница, если тул очень большой).
- Строка в **[wiki/TOOLS.md](../TOOLS.md)**.
- Обновить **[wiki/README.md](../README.md)** при новом разделе.

## 9. PR

- Код + документация wiki в **одном** PR (или сразу следующий коммит в ту же ветку), см. [CONTRIBUTING.md](../../CONTRIBUTING.md).
