"""Per-mode system prompt fragments for Lorne v1.0.

Prompts are loaded from ``Agent/prompts/<mode>.md`` files at runtime and cached.
If the file is missing the built-in string fallback is used (backward compat).
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

_NEW_TOOLS_HINT = (
    "\n**Новые инструменты (v1.0):** `task_decompose` — декомпозиция задачи; "
    "`structured_memory` — KV-память сессии; `multi_read` — читать 8 файлов за раз; "
    "`ast_analyze` — структура кода без чтения файла; `lint_check` — линтер после правок; "
    "`verify_result` — проверка условий задачи; `batch_replace` — пакетная замена; "
    "`env_info` — окружение; `session_notes` — свободные заметки."
)

_MODE_ADDONS_FALLBACK: Dict[str, str] = {
    "agent": (
        "### Режим Agent\n"
        "Полный цикл с тулами. Дисциплина: перед сложной задачей — `task_decompose`; "
        "`list_files`/`search_in_files` → `read_file` / `multi_read` → `ast_analyze` для структуры → "
        "`rag_search` для архитектуры → `plan_tool` на многошаговые задачи → правки "
        "(`replace_file_lines`/`batch_replace`/`write_file`/`code_file_tool`) → "
        "`lint_check` после правок → `verify_result` перед отчётом; "
        "факты → `structured_memory`.\n"
        + _JSON_IN_TOOLS + _NEW_TOOLS_HINT
    ),
    "ask": (
        "### Режим Ask\n"
        "Доступны только чтение и поиск: `list_files`, `read_file`, `read_file_lines`, "
        "`multi_read`, `ast_analyze`, `search_in_files`, `find_in_file`, `rag_search`, "
        "`web_search`, `web_fetch`, `library_context`, `get_file_line_count`, `ask_user`, "
        "`reasoning_tool` (think/analyze), `structured_memory` (get/list only), "
        "`env_info`, `session_notes` (read only), `ocr_tool`, `office_document_read` — "
        "**без** записи в файлы, без `run_command`, `edit_file`, `write_file`.\n"
        + _JSON_IN_TOOLS
    ),
    "creator": (
        "### Режим Creator\n"
        "Параллельные воркеры: каждый воркер использует `structured_memory` со своим namespace "
        "(worker_id) для изоляции состояния; `lint_check` + `verify_result` обязательны перед отчётом; "
        "`multi_read` для понимания кода. Формулируй цель и критерии готовности; веди `plan_tool`.\n"
        + _JSON_IN_TOOLS + _NEW_TOOLS_HINT
    ),
    "research": (
        "### Режим Research\n"
        "Опора на внешние источники: `web_search` → `web_fetch` для деталей; `library_context` "
        "для версий API пакетов; при связи с кодом репозитория — `rag_search` и `multi_read`; "
        "фиксируй находки в `session_notes` с тегом 'research'; `env_info` для проверки окружения.\n"
        + _JSON_IN_TOOLS + _NEW_TOOLS_HINT
    ),
    "deep": (
        "### Режим Deep\n"
        "Длинный автономный цикл: часто `plan_tool` + `reasoning_tool`; "
        "верифицированные факты → `structured_memory` (используй как лэджер); "
        "ход мыслей → `session_notes` с тегом 'milestone'; "
        "перед финалом — `verify_result` для проверки всех условий; избегай повторов; чекпоинты — по UI.\n"
        + _JSON_IN_TOOLS + _NEW_TOOLS_HINT
    ),
    "brainer": (
        "### Режим Brainer\n"
        "Сначала `rag_search` и файлы `project_brain/**`; `ast_analyze` + `multi_read` для быстрого "
        "обзора кода без чтения 500-строчных файлов; при устаревшем brain "
        "— `project_brain_tool` refresh, снова `rag_search`.\n"
        + _JSON_IN_TOOLS + _NEW_TOOLS_HINT
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
    return _load_mode_md(key)
