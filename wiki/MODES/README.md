# Режимы TUI

Каждый режим задаёт дополнительный фрагмент системного промпта — файл `Agent/prompts/<mode>.md` (фолбэк — `_MODE_ADDONS_FALLBACK` в `Agent/prompts/__init__.py`), возвращается через `mode_prompt_addon(mode)` — и влияет на набор тулов через `_sync_tui_tool_bundle` / `build_tools` в `Agent/agent/_impl_prepare.py`, а также на политику Project Brain (`Agent/project_brain/policy.py`: что индексировать/писать/дебаунсить в этом режиме).

| Режим | Файл | Кратко |
|-------|------|--------|
| Ask | [ask.md](ask.md) | Только чтение и поиск, без мутаций и без части тулов |
| Agent | [agent.md](agent.md) | Полный цикл, опционально браузер |
| Creator | [creator.md](creator.md) | Параллельные воркеры, те же туловые имена |
| Research | [research.md](research.md) | Акцент на веб и документацию пакетов |
| Deep | [deep.md](deep.md) | Отдельный долгий локальный цикл (не граф чата) |
| Brainer | [brainer.md](brainer.md) | Project Brain и RAG в приоритете |

**Редактор:** [Vi-like editor](../VI_EDITOR.md) · [Все клавиши](../KEYBINDINGS.md)

**Creator Mode:** роли воркеров с детальными промптами в `Agent/roles/` — Implementer, Reviewer, Researcher, Tester, Documenter, Integrator, Lead, Specialist.

Общий обзор UI: [TUI.md](../TUI.md). Архитектура: [ARCHITECTURE.md](../ARCHITECTURE.md).
