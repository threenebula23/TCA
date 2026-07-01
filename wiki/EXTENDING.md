# Расширение TCA

## Инструменты

Полный чеклист: **[developer/ADDING_TOOLS.md](developer/ADDING_TOOLS.md)**.

Кратко: реализация в `Agent/tools/` → реестр `Agent/tool_registry.py` → схемы `Agent/tool_schemas.py` → при необходимости `compact_tools.py` → UI `tool_card.py` → тесты → wiki ([TOOLS.md](TOOLS.md), [tool/REFERENCE.md](tool/REFERENCE.md)).

## TUI

Новые панели и стили: `Interface/panels/`, `tui_app.tcss`. Мост: только через `TUIBridge`. См. [Interface/EXTENDING.md](Interface/EXTENDING.md).

## Режимы и промпты

Фрагменты режимов: `Agent/prompts/<mode>.md` (фолбэк `_MODE_ADDONS_FALLBACK` в `Agent/prompts/__init__.py`). Общий системный текст: `Agent/system_prompt.py` (`SYSTEM_PROMPT` для сессии, `WORKER_SYSTEM_PROMPT` для воркеров Creator). Правила по Project Brain (общие для всех режимов): `Agent/prompts/project_brain_rules.py`; политика по режиму (что индексировать/писать/дебаунсить): `Agent/project_brain/policy.py`.

## Creator

Оркестрация и воркеры: `Agent/creator_mode.py`, `creator_orchestration.py`. Настройки: `orchestration_mode`, `orchestration_max_workers` в `ui_settings.json`.

Контракты: [developer/extension-contracts.md](developer/extension-contracts.md).
