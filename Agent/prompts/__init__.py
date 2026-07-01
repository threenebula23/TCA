"""Per-mode system prompt fragments for Lorne v1.0.

Prompts are loaded from ``Agent/prompts/<mode>.md`` files at runtime and cached.
If the file is missing the built-in string fallback is used (backward compat).

J4 (context budget): each mode file now holds only its *delta* — the workflow
specific to that mode. Two blocks that used to be copy-pasted into every
single mode file (``_JSON_IN_TOOLS`` and the old ``_NEW_TOOLS_HINT`` tool
laundry list) are appended/handled centrally here instead:

- ``_JSON_IN_TOOLS`` is still useful (models regularly break long JSON
  string args) so it's appended once by :func:`mode_prompt_addon`.
- The old tool-name list is dropped entirely — tool names/descriptions
  already live in the JSON schemas bound to the model, and a full listing
  is available on demand via the ``tools_catalog`` tool
  (``action=list``/``describe``) instead of being baked into every prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

_HERE = Path(__file__).parent

_JSON_IN_TOOLS = (
    "Строковые JSON-аргументы (`steps_json`, `data_json`, …): одна строка = **полный** "
    "валидный JSON в **двойных** кавычках, без обрыва; для `plan_tool` save надёжнее "
    "поле `steps` как массив строк."
)

_VIZ_HINT = (
    "Графики/диаграммы в ответе: вызови `viz_tool`, затем вставь его результат в финальный "
    "ответ как fenced-блок ```lorne-chart / ```mermaid (тело — JSON `chart` или текст "
    "`diagram` из результата тула) — TUI отрендерит их сам; голый JSON или Vega-Lite спека "
    "в тексте не отрисуются."
)

_MODE_ADDONS_FALLBACK: Dict[str, str] = {
    "agent": (
        "### Режим Agent\n"
        "Полный цикл с тулами. `list_files`/`search_in_files` → `read_file`/`multi_read` → "
        "`ast_analyze` для структуры → `rag_search` для архитектуры → `plan_tool` на "
        "многошаговые задачи → правки → `lint_check` → `verify_result` перед отчётом."
    ),
    "ask": (
        "### Режим Ask\n"
        "Только чтение и поиск — без записи в файлы, без `run_command`, `edit_file`, "
        "`write_file`, `project_brain_tool`. Для фиксации выводов в Project Brain "
        "переключись в Agent или Brainer."
    ),
    "creator": (
        "### Режим Creator\n"
        "Параллельные воркеры: каждый — свой namespace в `structured_memory` (worker_id); "
        "`lint_check` + `verify_result` обязательны перед отчётом. Формулируй цель и "
        "критерии готовности; веди `plan_tool`."
    ),
    "research": (
        "### Режим Research\n"
        "Опора на внешние источники: `web_search` → `web_fetch`; `library_context` для "
        "версий пакетов; при связи с кодом — `rag_search`. Важные находки — сразу "
        "`project_brain_tool` `write_brain` в `agent/research_notes.md` (не только "
        "`session_notes`, которые не переживают сессию)."
    ),
    "deep": (
        "### Режим Deep\n"
        "Длинный автономный цикл: `plan_tool` + `reasoning_tool`; верифицированные факты — "
        "в `structured_memory`; перед финалом — `verify_result`. Итоговый отчёт "
        "автоматически попадает в `agent/deep_report.md`."
    ),
    "brainer": (
        "### Режим Brainer — RAG-first, единственный режим с авто-обслуживанием brain\n"
        "**Workflow:** 1) `rag_search` (brain выше кода) → 2) при необходимости "
        "`read_file` по `project_brain/*.md` → 3) `ast_analyze`/`multi_read` по коду → "
        "4) правки → 5) устойчивые выводы — `project_brain_tool` `write_brain` "
        "(`brain_rel_path` в `agent/*.md`).\n\n"
        "**Когда писать в brain** (а не только отвечать в чате): архитектурные решения, "
        "контракты API, найденные риски/баги, зависимости между модулями — то, что "
        "полезно в *следующей* сессии. Не дублируй сиюминутные детали хода.\n\n"
        "**Перед финальным ответом без вызова тулов** — если за ход не было ни одного "
        "`write_brain`, сделай короткую запись в `agent/session_notes.md` с сутью "
        "находок; полный пересбор (`action=refresh`) сканер делает сам после хода — "
        "вызывай `refresh` вручную только если структура репозитория заметно изменилась."
    ),
}

_file_cache: Dict[str, str] = {}


def _load_mode_md(mode: str) -> str:
    """Load prompt fragment from Agent/prompts/<mode>.md, cache result."""
    if mode in _file_cache:
        return _file_cache[mode]
    md_path = _HERE / f"{mode}.md"
    if md_path.is_file():
        try:
            text = md_path.read_text(encoding="utf-8").strip()
            _file_cache[mode] = text
            return text
        except Exception:
            pass
    fallback = _MODE_ADDONS_FALLBACK.get(mode, "")
    _file_cache[mode] = fallback
    return fallback


def mode_prompt_addon(mode: str) -> str:
    """Return system prompt fragment for *mode* slug, or empty string."""
    key = (mode or "").strip().lower()
    body = _load_mode_md(key)
    if not body:
        return body
    suffix = _JSON_IN_TOOLS
    try:
        from Agent.tool_registry import _tool_session_flags

        if _tool_session_flags.get("extended", False):
            suffix += "\n\n" + _VIZ_HINT
    except Exception:
        pass
    return body + "\n\n" + suffix
