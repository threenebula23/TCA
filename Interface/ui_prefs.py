"""Настройки UI (тема, плотность, синтаксис) в каталоге данных проекта (``.lorne`` / legacy ``.tca``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_PREFS: Dict[str, Any] = {
    "theme": "Purple Dark",
    # Классический CLI: id пресетов Interface.cli_theme.CLI_THEME_PALETTES (purple, ocean, …).
    "cli_theme": "purple",
    "cli_accent_color": "#8B5CF6",
    "density": "normal",
    "syntax_theme": "monokai",
    "accent_color": "#8B5CF6",
    # В режиме Agent: подключать Python Playwright (Chromium), если True — см. Settings.
    "playwright_python_enabled": False,
    # В режиме Agent: включать браузерные Node-tools (headless browser layer).
    "browser_tools_enabled": True,
    # Пользовательские модели для селектора (хранятся в проекте).
    "openrouter_custom_models": [],
    "ollama_custom_models": [],
    "lmstudio_custom_models": [],
    # Настройки подключения Ollama.
    "ollama_base_url": "http://localhost:11434/v1",
    "ollama_api_key": "",
    # Настройки подключения LM Studio (OpenAI-compatible /v1 only).
    "lmstudio_base_url": "http://localhost:1234/v1",
    "lmstudio_api_key": "",
    "ollama_presets": {
        "default": {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            # 1.1-1.15 is the sweet spot recommended for local models to stop
            # them looping/repeating text without over-penalizing legitimate
            # repetition (e.g. code keywords). Paired with repeat_last_n below.
            "repeat_penalty": 1.15,
            # How many recent tokens the repeat penalty looks back over. Too
            # small (e.g. Ollama's old internal default of 64) lets a model
            # fall into paragraph-level loops; 256 covers a few sentences.
            "repeat_last_n": 256,
            "num_ctx": 32768,
            # 2048 routinely truncated long code/explanation answers mid-way
            # (looked like the model "hanging" or stopping abruptly). 8192
            # matches the "balanced" profile's max_tokens default instead.
            "num_predict": 8192,
            "stop": "",
        }
    },
    "ollama_model_settings": {},
    # Creator orchestration (parallel | pipeline | auto).
    "orchestration_mode": "auto",
    # Max parallel workers when creator runs in parallel orchestration.
    "orchestration_max_workers": 4,
    # Research mode knobs — both apply to local + remote.
    "research_max_sources": 6,
    "research_max_rounds": 3,
    "research_deep_fetch": True,
    # Glyph for classic CLI / prompt_toolkit line (Rich markup stripped on save).
    "cli_prompt_glyph": "❯",
    # Custom tools master switch (RAG, planning, interpreter, thinking, etc.).
    "custom_tools_enabled": True,
    # Extended mega-tools (code_intel_tool, workspace_search, net_tool, viz_tool, …,
    # see Agent/tools/extended_tools.py). Off by default (J1, context budget):
    # binding them alongside the full base tool set roughly doubles the JSON
    # schema payload sent with every LLM call; opt in for power users who want
    # find_symbol/import_graph/http/diff/apply_patch/etc. in the tool list.
    "extended_tools_enabled": False,
}


def prefs_path() -> Path:
    try:
        from Agent.path_utils import get_project_root
        from Agent.runtime_paths import project_data_dir

        root = project_data_dir(get_project_root())
    except Exception:
        from Agent.runtime_paths import project_data_dir

        root = project_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / "ui_settings.json"


def load_prefs() -> Dict[str, Any]:
    p = prefs_path()
    if not p.exists():
        return dict(DEFAULT_PREFS)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        out = dict(DEFAULT_PREFS)
        out.update({k: v for k, v in data.items() if k in DEFAULT_PREFS})
        if "cli_theme" not in data and data.get("theme"):
            from Interface.cli_theme import resolve_cli_theme_name

            out["cli_theme"] = resolve_cli_theme_name(str(data.get("theme")))
        if "cli_accent_color" not in data and data.get("accent_color"):
            out["cli_accent_color"] = str(data["accent_color"])
        return out
    except Exception:
        return dict(DEFAULT_PREFS)


def cli_prompt_prefix_plain() -> str:
    """Plain-text CLI prompt prefix (safe for Rich); ends with a space."""
    try:
        g = str(load_prefs().get("cli_prompt_glyph") or "❯").strip() or "❯"
    except Exception:
        g = "❯"
    g = g.replace("[", "").replace("]", "")[:12]
    return g + (" " if not g.endswith(" ") else "")


def save_prefs(**kwargs: Any) -> None:
    current = load_prefs()
    for k, v in kwargs.items():
        if k in DEFAULT_PREFS:
            current[k] = v
    try:
        prefs_path().write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
