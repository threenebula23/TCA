"""
Провайдер LLM для Lorne: профили, сохранение конфигурации, OpenRouter.
Включая определение возможностей провайдера для совместимых параметров API.
"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from Agent.runtime_paths import env_pref, project_data_dir, user_config_json_path

try:  # Preferred native client for Ollama — forwards all options to /api/chat.
    from langchain_ollama import ChatOllama  # type: ignore

    _HAS_CHAT_OLLAMA = True
except Exception:  # pragma: no cover — package is optional at runtime.
    ChatOllama = None  # type: ignore
    _HAS_CHAT_OLLAMA = False

ProfileName = str

# ─── Provider capabilities ──────────────────────────────────────────
# Maps provider prefix → supported features.
# parallel_tool_calls: safe to pass parallel_tool_calls=False to bind_tools
#   (OpenAI natively supports it; others may reject the extra key via OpenRouter)
_PROVIDER_CAPS: Dict[str, Dict[str, bool]] = {
    "openai/":      {"parallel_tool_calls": True,  "native_tools": True},
    "anthropic/":   {"parallel_tool_calls": False, "native_tools": True},
    "meta-llama/":  {"parallel_tool_calls": False, "native_tools": True},
    "deepseek/":    {"parallel_tool_calls": False, "native_tools": True},
    "google/":      {"parallel_tool_calls": False, "native_tools": True},
    "qwen/":        {"parallel_tool_calls": False, "native_tools": True},
    "mistralai/":   {"parallel_tool_calls": False, "native_tools": True},
    "arcee-ai/":    {"parallel_tool_calls": False, "native_tools": True},
    "stepfun/":     {"parallel_tool_calls": False, "native_tools": True},
    "x-ai/":        {"parallel_tool_calls": False, "native_tools": True},
    # Local Ollama models: served via native /api/chat (ChatOllama) or /v1 (ChatOpenAI).
    # Native tool-calling support varies by model family (qwen2.5, llama3.1, mistral-v0.3, …),
    # but our recovery layer in message_utils handles non-native models by extracting
    # tool calls from plain-text responses.
    "ollama/":      {"parallel_tool_calls": False, "native_tools": True},
    # LM Studio: OpenAI-compatible /v1 server only (no native endpoint like
    # Ollama's /api/chat). Function-calling support depends on the loaded
    # model, same caveat as Ollama; the message_utils recovery layer applies
    # here too since both speak the same OpenAI tool-call wire format.
    "lmstudio/":    {"parallel_tool_calls": False, "native_tools": True},
}


# ─── Ollama per-tag quirks registry ─────────────────────────────────
# `_PROVIDER_CAPS["ollama/"]` above is a single flag set shared by *every*
# Ollama tag, even though native tool-calling support, safe context size and
# stop tokens vary a lot by model family. This registry lets us apply
# family-specific defaults instead of guessing the same thing for all of
# them — it's the concrete reason "some Ollama models don't work right".
# Matched by substring against the lowercased wire model name (e.g. "qwen2.5:7b").
_OLLAMA_MODEL_QUIRKS: List[Dict[str, Any]] = [
    {"match": ("qwen2.5", "qwen2"), "supports_tools": True, "default_num_ctx": 32768, "stop": []},
    {"match": ("qwen3",), "supports_tools": True, "default_num_ctx": 40960, "stop": []},
    {"match": ("llama3.1", "llama-3.1"), "supports_tools": True, "default_num_ctx": 131072, "stop": ["<|eot_id|>"]},
    {"match": ("llama3.2", "llama-3.2"), "supports_tools": True, "default_num_ctx": 131072, "stop": ["<|eot_id|>"]},
    {"match": ("llama3", "llama-3"), "supports_tools": False, "default_num_ctx": 8192, "stop": ["<|eot_id|>"]},
    {"match": ("mistral-nemo",), "supports_tools": True, "default_num_ctx": 131072, "stop": []},
    {"match": ("mistral",), "supports_tools": True, "default_num_ctx": 32768, "stop": []},
    {"match": ("mixtral",), "supports_tools": True, "default_num_ctx": 32768, "stop": []},
    {"match": ("gemma2",), "supports_tools": False, "default_num_ctx": 8192, "stop": ["<end_of_turn>"]},
    {"match": ("gemma3", "gemma"), "supports_tools": False, "default_num_ctx": 32768, "stop": ["<end_of_turn>"]},
    {"match": ("phi3", "phi-3"), "supports_tools": False, "default_num_ctx": 4096, "stop": ["<|end|>"]},
    {"match": ("phi4", "phi-4"), "supports_tools": True, "default_num_ctx": 16384, "stop": ["<|end|>"]},
    {"match": ("deepseek-r1",), "supports_tools": False, "default_num_ctx": 32768, "stop": []},
    {"match": ("deepseek-coder", "deepseek-v"), "supports_tools": True, "default_num_ctx": 32768, "stop": []},
    {"match": ("codellama",), "supports_tools": False, "default_num_ctx": 16384, "stop": []},
]
_OLLAMA_DEFAULT_QUIRK: Dict[str, Any] = {"supports_tools": True, "default_num_ctx": 32768, "stop": []}


def get_ollama_model_quirks(wire_model_name: str) -> Dict[str, Any]:
    """Best-effort per-family defaults for an Ollama tag (e.g. ``qwen2.5:7b``).

    Falls back to a generic "assume native tools, 32k ctx" entry for unknown
    families. This does not replace explicit user overrides in
    ``ollama_model_settings`` — it only fills in sane defaults so unfamiliar
    tags don't silently inherit settings tuned for a different model.
    """
    name = (wire_model_name or "").lower()
    for entry in _OLLAMA_MODEL_QUIRKS:
        if any(tag in name for tag in entry["match"]):
            return dict(entry)
    return dict(_OLLAMA_DEFAULT_QUIRK)


# Cache of `/api/show` lookups so we don't hit the local server on every LLM
# call just to read the model's real context window.
_MODEL_SHOW_CACHE: Dict[str, Dict[str, Any]] = {}
_MODEL_SHOW_CACHE_TTL = 300.0


def fetch_ollama_model_show(
    wire_model_name: str, base_url: str = "", api_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Query Ollama ``/api/show`` for a model's real metadata (cached).

    Used to clip a too-large ``num_ctx`` down to what the model actually
    supports, instead of trusting a single hardcoded default for every tag.
    """
    cache_key = f"{base_url}::{wire_model_name}"
    cached = _MODEL_SHOW_CACHE.get(cache_key)
    now = time.time()
    if cached and (now - cached.get("_fetched_at", 0)) < _MODEL_SHOW_CACHE_TTL:
        return cached.get("data")

    data = _ollama_http_json(
        "POST", "/api/show", payload={"model": wire_model_name},
        base_url=base_url, api_key=api_key, timeout=6,
    )
    _MODEL_SHOW_CACHE[cache_key] = {"_fetched_at": now, "data": data}
    return data


def get_ollama_model_max_ctx(wire_model_name: str, base_url: str = "", api_key: str = "") -> Optional[int]:
    """Best-effort real context-length ceiling for a model, via ``/api/show``.

    Returns ``None`` when unknown (offline server, unsupported field, etc.)
    so callers can fall back to the family-quirk default instead.
    """
    info = fetch_ollama_model_show(wire_model_name, base_url=base_url, api_key=api_key)
    if not isinstance(info, dict):
        return None
    model_info = info.get("model_info") if isinstance(info.get("model_info"), dict) else {}
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            try:
                n = int(value)
                if n > 0:
                    return n
            except Exception:
                continue
    return None


def get_model_capabilities(model_id: str) -> Dict[str, bool]:
    """Return capability flags for a given model based on its provider prefix.

    For ``ollama/<tag>`` models this refines the generic prefix-level flags
    with the per-tag quirks registry, since native tool-calling support is
    not uniform across local model families (see ``_OLLAMA_MODEL_QUIRKS``).
    """
    mid = model_id or ""
    if mid.startswith("ollama/"):
        caps = dict(_PROVIDER_CAPS["ollama/"])
        wire_name = mid.split("/", 1)[1]
        caps["native_tools"] = bool(get_ollama_model_quirks(wire_name).get("supports_tools", True))
        return caps
    for prefix, caps in _PROVIDER_CAPS.items():
        if mid.startswith(prefix):
            return dict(caps)
    return {"parallel_tool_calls": False, "native_tools": True}


def supports_parallel_tool_calls_param(model_id: str) -> bool:
    """Check if the model's provider supports the parallel_tool_calls binding parameter."""
    return get_model_capabilities(model_id).get("parallel_tool_calls", False)


def is_reasoning_model(model_id: str) -> bool:
    """Check if the model is a reasoning/thinking model that emits <think> blocks."""
    _id = (model_id or "").lower()
    return any(tag in _id for tag in ("deepseek-r1", "qwq", "/o1", "/o3", "/o4"))


# ─── Popular OpenRouter models (curated) ───────────────────────────
AVAILABLE_MODELS: List[Dict[str, Any]] = [
    # --- free ---
    {"id": "arcee-ai/trinity-large-preview:free",       "name": "Trinity Large",        "ctx": 131_072,    "tier": "free"},
    {"id": "stepfun/step-3.5-flash:free",               "name": "Step 3.5 Flash",        "ctx": 256_000,    "tier": "free"},
    {"id": "qwen/qwen3-235b-a22b-thinking-2507",         "name": "Qwen3 235B Thinking",   "ctx": 131_072,    "tier": "free"},
    # --- paid ---
    {"id": "qwen/qwen3-coder-30b-a3b-instruct",         "name": "Qwen3 Coder 30B",       "ctx": 160_000,    "tier": "paid"},
    {"id": "qwen/qwen3.5-flash-02-23",                  "name": "Qwen3.5 Flash",        "ctx": 1_000_000,    "tier": "paid"},
    {"id": "openai/gpt-oss-120b",                       "name": "GPT OSS 120B",          "ctx": 131_072,    "tier": "paid"},
    {"id": "openai/gpt-5-nano",                         "name": "GPT-5 Nano",             "ctx": 400_000,    "tier": "paid"},
    {"id": "google/gemini-2.5-flash-lite",              "name": "Gemini 2.5 Flash Lite", "ctx": 1_048_576,  "tier": "paid"},
    # --- cheap ---
    {"id": "qwen/qwen3-coder-next",                     "name": "Qwen3 Coder Next",     "ctx": 262_144,    "tier": "cheap"},
    {"id": "qwen/qwen3.5-35b-a3b",                      "name": "Qwen3.5 35B",           "ctx": 262_144,    "tier": "cheap"},
    {"id": "qwen/qwen3.5-plus-02-15",                   "name": "Qwen3.5 Plus",         "ctx": 1_000_000,    "tier": "cheap"},
    {"id": "qwen/qwen3.5-397b-a17b",                    "name": "Qwen3.5 397B",         "ctx": 262_144,    "tier": "cheap"},
    {"id": "openai/gpt-4o-mini",                        "name": "GPT-4o Mini",           "ctx": 128_000,    "tier": "cheap"},
    {"id": "openai/gpt-5-mini",                         "name": "GPT-5 Mini",            "ctx": 400_000,    "tier": "cheap"},
    {"id": "openai/gpt-5.1-codex-mini",                 "name": "GPT-5.1 Codex Mini",     "ctx": 400_000,    "tier": "cheap"},
    {"id": "google/gemini-2.5-flash",                   "name": "Gemini 2.5 Flash",      "ctx": 1_048_576,  "tier": "cheap"},
    {"id": "google/gemini-3-flash-preview",             "name": "Gemini 3 Flash",        "ctx": 1_048_576,  "tier": "cheap"},
    {"id": "x-ai/grok-4.1-fast",                        "name": "Grok 4.1 Fast",         "ctx": 2_000_000,    "tier": "cheap"},
    {"id": "x-ai/grok-code-fast-1",                     "name": "Grok Code Fast",       "ctx": 256_000,    "tier": "cheap"},
    {"id": "deepseek/deepseek-v3.2",                     "name": "DeepSeek V3.2",         "ctx": 163_840,    "tier": "cheap"},
    # --- pro ---
    {"id": "openai/gpt-5.1-codex-max",                  "name": "GPT-5.1 Codex Max",     "ctx": 400_000,    "tier": "pro"},
    {"id": "openai/gpt-5.3-codex",                      "name": "GPT-5.3 Codex",         "ctx": 400_000,    "tier": "pro"},
    {"id": "google/gemini-3.1-pro-preview",             "name": "Gemini 3.1 Pro",         "ctx": 1_048_576,  "tier": "pro"},
    {"id": "anthropic/claude-haiku-4.5",                "name": "Claude Haiku 4.5",       "ctx": 200_000,    "tier": "pro"},
    {"id": "anthropic/claude-sonnet-4.6",               "name": "Claude Sonnet 4.6",     "ctx": 1_000_000,    "tier": "pro"},
    {"id": "anthropic/claude-opus-4.6",                 "name": "Claude Opus 4.6",       "ctx": 1_000_000,    "tier": "pro"},
]


def _ensure_v1_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return "http://localhost:11434/v1"
    if u.endswith("/v1"):
        return u
    return u + "/v1"


def _parse_stop_sequences(raw: Any) -> List[str]:
    """Split a stop-sequence spec entered via UI/env into a clean list.

    Priority of separators: already-a-list > newlines > commas. We can't use
    `|` because real stop tokens contain it (e.g. `<|im_end|>`, `<|eot_id|>`).
    Duplicates and empty strings are dropped; order is preserved.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        text = str(raw).strip()
        if not text:
            return []
        if "\n" in text:
            items = [s.strip() for s in text.splitlines() if s.strip()]
        elif "," in text:
            items = [s.strip() for s in text.split(",") if s.strip()]
        else:
            items = [text]

    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _ensure_native_base_url(url: str) -> str:
    """Strip the trailing /v1 so ChatOllama hits native /api/chat directly."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return "http://localhost:11434"
    if u.endswith("/v1"):
        return u[:-3].rstrip("/") or "http://localhost:11434"
    return u


def _load_ui_model_overrides() -> Dict[str, Any]:
    """Читает переопределения моделей из ``<project>/.lorne/ui_settings.json`` (или legacy ``.tca``)."""
    p = project_data_dir() / "ui_settings.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text("utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _dedupe_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for m in models:
        mid = str(m.get("id") or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(m)
    return out


def get_available_models() -> List[Dict[str, Any]]:
    """Curated models + user-added OpenRouter/Ollama models from UI prefs."""
    prefs = _load_ui_model_overrides()
    merged: List[Dict[str, Any]] = list(AVAILABLE_MODELS)

    for m in (prefs.get("openrouter_custom_models") or []):
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        merged.append(
            {
                "id": mid,
                "name": str(m.get("name") or mid),
                "ctx": int(m.get("ctx") or 128_000),
                "tier": str(m.get("tier") or "custom"),
                "source": "openrouter",
            }
        )

    for m in (prefs.get("ollama_custom_models") or []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        merged.append(
            {
                "id": f"ollama/{name}",
                "name": str(m.get("label") or f"Ollama · {name}"),
                "ctx": int(m.get("ctx") or 32_768),
                "tier": "local",
                "source": "ollama",
            }
        )

    for m in (prefs.get("lmstudio_custom_models") or []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        merged.append(
            {
                "id": f"lmstudio/{name}",
                "name": str(m.get("label") or f"LM Studio · {name}"),
                "ctx": int(m.get("ctx") or 32_768),
                "tier": "local",
                "source": "lmstudio",
            }
        )

    return _dedupe_models(merged)


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default or ""
    return value


# ─── Config persistence ────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    try:
        p = user_config_json_path()
        if p.exists():
            return json.loads(p.read_text("utf-8"))
    except Exception:
        pass
    return {}


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        existing = load_config()
        existing.update(cfg)
        user_config_json_path().write_text(json.dumps(existing, indent=2, ensure_ascii=False), "utf-8")
    except Exception:
        pass


def get_saved_model() -> Optional[str]:
    return load_config().get("model")


def save_model_choice(model_id: str) -> None:
    save_config({"model": model_id})


# ─── Profiles ──────────────────────────────────────────────────────

def _resolve_default_model() -> str:
    """Priority: env var > saved config > hardcoded default."""
    env_model = env_pref("MODEL")
    if env_model:
        return env_model
    saved = get_saved_model()
    if saved:
        return saved
    return "arcee-ai/trinity-large-preview:free"


def _build_profiles() -> Dict[ProfileName, Dict[str, object]]:
    base_model = _resolve_default_model()
    return {
        "fast": {
            "model": env_pref("MODEL_FAST", base_model),
            "temperature": float(env_pref("TEMP_FAST", "0.1")),
            "max_tokens": int(env_pref("MAX_TOKENS_FAST", env_pref("MAX_TOKENS", "4096"))),
        },
        "balanced": {
            "model": env_pref("MODEL_BALANCED", base_model),
            "temperature": float(env_pref("TEMP_BALANCED", "0.2")),
            "max_tokens": int(env_pref("MAX_TOKENS_BALANCED", env_pref("MAX_TOKENS", "8192"))),
        },
        "quality": {
            "model": env_pref("MODEL_QUALITY", base_model),
            "temperature": float(env_pref("TEMP_QUALITY", "0.1")),
            "max_tokens": int(env_pref("MAX_TOKENS_QUALITY", env_pref("MAX_TOKENS", "16384"))),
        },
    }


_PROFILES: Dict[ProfileName, Dict[str, object]] = _build_profiles()


def reload_profiles() -> None:
    """Rebuild profiles after model change."""
    global _PROFILES
    _PROFILES = _build_profiles()


def get_available_profiles() -> Dict[ProfileName, Dict[str, object]]:
    return dict(_PROFILES)


def normalize_profile(name: str | None) -> ProfileName:
    if not name:
        env_profile = env_pref("PROFILE", "balanced").lower()
        name = env_profile or "balanced"
    name = name.lower().strip()
    if name in _PROFILES:
        return name
    aliases = {
        "f": "fast", "fast-profile": "fast",
        "q": "quality", "quality-profile": "quality", "hi": "quality", "high": "quality",
        "b": "balanced", "balanced-profile": "balanced", "mid": "balanced", "medium": "balanced",
    }
    return aliases.get(name, "balanced")


def _resolve_ollama_settings(
    wire_model_name: str,
    base_temperature: float,
    base_max_tokens: int,
) -> Dict[str, Any]:
    """Return merged Ollama settings (url/key/generation params) for a model.

    Layered defaults → preset → per-model overrides. Empty/None values never
    override a defined layer. Suitable for both ChatOllama and the ChatOpenAI
    fallback (caller picks which keys it can forward).
    """
    prefs = _load_ui_model_overrides()

    base_url_raw = _env(
        "OLLAMA_BASE_URL",
        str(prefs.get("ollama_base_url") or _env("LOCAL_MODEL_URL", "http://localhost:11434/v1")),
    )
    api_key = (
        _env("OLLAMA_API_KEY", str(prefs.get("ollama_api_key") or _env("LOCAL_MODEL_API_KEY", "ollama")))
        or "ollama"
    )

    presets = prefs.get("ollama_presets") if isinstance(prefs.get("ollama_presets"), dict) else {}
    model_map = (
        prefs.get("ollama_model_settings")
        if isinstance(prefs.get("ollama_model_settings"), dict)
        else {}
    )
    raw_cfg = model_map.get(wire_model_name) if isinstance(model_map.get(wire_model_name), dict) else {}
    preset_name = str(raw_cfg.get("preset") or "default")
    preset_cfg = presets.get(preset_name) if isinstance(presets.get(preset_name), dict) else {}

    # num_predict defaults: fall back on the profile's max_tokens so coding
    # replies aren't truncated at the old 2048 cap mid-answer. 0 or -1 means
    # "no limit" in Ollama — pass them through untouched.
    try:
        mp_default = int(base_max_tokens) if int(base_max_tokens or 0) > 0 else 8192
    except Exception:
        mp_default = 8192
    quirks = get_ollama_model_quirks(wire_model_name)
    merged: Dict[str, Any] = {
        "temperature": base_temperature,
        "top_p": 0.9,
        "top_k": 40,
        # 1.1-1.15 discourages local models from looping/repeating text
        # without over-penalizing legitimate repetition; repeat_last_n
        # controls how far back that penalty looks (too small lets a model
        # fall into paragraph-level loops).
        "repeat_penalty": 1.15,
        "repeat_last_n": 256,
        "num_ctx": quirks.get("default_num_ctx", 32768),
        "num_predict": mp_default,
        "stop": quirks.get("stop") or "",
    }
    merged.update({k: v for k, v in preset_cfg.items() if v is not None})
    merged.update({k: v for k, v in raw_cfg.items() if k != "preset" and v is not None})

    stop_list = _parse_stop_sequences(merged.get("stop"))

    # Clip num_ctx to the model's real context window when we can find it out
    # (via /api/show) — a too-large num_ctx either gets silently truncated
    # server-side or, on constrained hardware, can blow up VRAM/RAM usage.
    requested_num_ctx = int(merged.get("num_ctx", quirks.get("default_num_ctx", 32768)) or 32768)
    try:
        real_max_ctx = get_ollama_model_max_ctx(
            wire_model_name,
            base_url=_ensure_native_base_url(base_url_raw),
            api_key=api_key,
        )
    except Exception:
        real_max_ctx = None
    if real_max_ctx and requested_num_ctx > real_max_ctx:
        requested_num_ctx = real_max_ctx

    try:
        num_predict_final = int(merged.get("num_predict", mp_default))
    except Exception:
        num_predict_final = mp_default
    # Clamp tiny values that obviously truncate the answer (≤128 tokens).
    if 0 < num_predict_final < 128:
        num_predict_final = mp_default

    return {
        "base_url_v1": _ensure_v1_base_url(base_url_raw),
        "base_url_native": _ensure_native_base_url(base_url_raw),
        "api_key": api_key,
        "temperature": float(merged.get("temperature", base_temperature)),
        "top_p": float(merged.get("top_p", 0.9)),
        "top_k": int(merged.get("top_k", 40)),
        "repeat_penalty": float(merged.get("repeat_penalty", 1.15)),
        "repeat_last_n": int(merged.get("repeat_last_n", 256)),
        "num_ctx": requested_num_ctx,
        "num_predict": num_predict_final,
        "stop": stop_list,
        "supports_tools": bool(quirks.get("supports_tools", True)),
    }


def _build_ollama_chat_llm(wire_model_name: str, settings: Dict[str, Any]) -> Any:
    """Instantiate ChatOllama with all Ollama-native options applied."""
    if not _HAS_CHAT_OLLAMA or ChatOllama is None:  # defensive
        raise RuntimeError("langchain-ollama is not installed")

    kwargs: Dict[str, Any] = {
        "model": wire_model_name,
        "base_url": settings["base_url_native"],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "top_k": settings["top_k"],
        "repeat_penalty": settings["repeat_penalty"],
        "repeat_last_n": settings.get("repeat_last_n", 256),
        "num_ctx": settings["num_ctx"],
        "num_predict": settings["num_predict"],
        # Local models can be slow on first load — give them breathing room.
        "client_kwargs": {"timeout": 600},
    }
    if settings.get("stop"):
        kwargs["stop"] = list(settings["stop"])
    return ChatOllama(**kwargs)


def _build_ollama_openai_llm(wire_model_name: str, settings: Dict[str, Any]) -> ChatOpenAI:
    """Fallback: Ollama served through its OpenAI-compat /v1 endpoint.

    Only OpenAI-standard parameters can be forwarded reliably; Ollama-native
    options (num_ctx, top_k, repeat_penalty) are NOT honoured by /v1 and are
    skipped instead of silently misleading users through a dead extra_body.
    """
    # Ollama's `num_predict=-1` / 0 mean "no limit"; /v1 rejects that, so
    # fall back to a large but finite cap instead of propagating the sentinel.
    np = int(settings.get("num_predict", 8192))
    max_tokens = np if np > 0 else 8192
    kwargs: Dict[str, Any] = {
        "base_url": settings["base_url_v1"],
        "api_key": settings["api_key"] or "ollama",
        "model": wire_model_name,
        "temperature": settings["temperature"],
        "max_tokens": max_tokens,
        "top_p": settings["top_p"],
        # 10 min is a sensible upper bound for long local generations.
        "request_timeout": 600,
        # No retries — local servers either respond or they don't.
        "max_retries": 1,
    }
    if settings.get("stop"):
        # ChatOpenAI accepts `stop` as a top-level kwarg; passing via model_kwargs
        # triggers a warning and the field gets stripped by the validator.
        kwargs["stop"] = list(settings["stop"])
    return ChatOpenAI(**kwargs)


def _resolve_lmstudio_settings(base_temperature: float, base_max_tokens: int) -> Dict[str, Any]:
    """Resolve LM Studio connection settings (base_url/api_key) + sampling params.

    LM Studio only exposes an OpenAI-compatible ``/v1`` server (default port
    1234), so unlike Ollama there's no native endpoint to fall back from —
    every request goes through ``ChatOpenAI``.
    """
    prefs = _load_ui_model_overrides()
    base_url_raw = _env(
        "LMSTUDIO_BASE_URL",
        str(prefs.get("lmstudio_base_url") or "http://localhost:1234/v1"),
    )
    api_key = (
        _env("LMSTUDIO_API_KEY", str(prefs.get("lmstudio_api_key") or "")) or "lm-studio"
    )
    try:
        mp_default = int(base_max_tokens) if int(base_max_tokens or 0) > 0 else 8192
    except Exception:
        mp_default = 8192
    return {
        "base_url_v1": _ensure_v1_base_url(base_url_raw),
        "api_key": api_key,
        "temperature": base_temperature,
        "top_p": 0.9,
        "max_tokens": mp_default,
    }


def _build_lmstudio_llm(wire_model_name: str, settings: Dict[str, Any]) -> ChatOpenAI:
    """LM Studio is OpenAI-compatible; reuse the same client as OpenRouter/Ollama-/v1."""
    return ChatOpenAI(
        base_url=settings["base_url_v1"],
        api_key=settings["api_key"] or "lm-studio",
        model=wire_model_name,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        top_p=settings["top_p"],
        request_timeout=600,
        max_retries=1,
    )


def _lmstudio_native_base(base_url_v1: str) -> str:
    """Turn the OpenAI-compatible ``/v1`` base into LM Studio's native ``/api/v0``.

    The native endpoint exposes real per-model context length (``GET /v1/models``
    only returns bare ids), so it's needed to avoid showing a fake hardcoded ctx.
    """
    u = (base_url_v1 or "").rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    return u + "/api/v0"


def fetch_lmstudio_models(base_url: str = "", api_key: str = "") -> List[Dict[str, Any]]:
    """Fetch model list with real context length from LM Studio's native API.

    Tries ``GET /api/v0/models`` first (LM Studio-specific; reports
    ``max_context_length`` / ``loaded_context_length`` per model and whether a
    model is actually loaded). Falls back to the bare OpenAI-compatible
    ``GET /v1/models`` (no ctx info, so we fall back to a conservative default)
    if the native endpoint isn't available (older LM Studio versions).
    """
    raw = (base_url or "").strip() or _env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    base_v1 = _ensure_v1_base_url(raw)
    token = (api_key or "").strip() or _env("LMSTUDIO_API_KEY", "")
    out: List[Dict[str, Any]] = []

    native_base = _lmstudio_native_base(base_v1)
    req = urllib.request.Request(native_base + "/models", method="GET")
    req.add_header("Authorization", f"Bearer {token or 'lm-studio'}")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in (data.get("data") or []):
            if not isinstance(m, dict):
                continue
            name = str(m.get("id") or "").strip()
            if not name:
                continue
            # Embeddings / reranker / TTS models aren't chat models — only
            # "llm" entries are valid completions targets.
            mtype = str(m.get("type") or "llm").strip().lower()
            if mtype and mtype != "llm":
                continue
            loaded = str(m.get("state") or "").strip().lower() == "loaded"
            ctx = m.get("loaded_context_length") if loaded else None
            if not isinstance(ctx, int) or ctx <= 0:
                ctx = m.get("max_context_length")
            if not isinstance(ctx, int) or ctx <= 0:
                ctx = 32_768
            out.append({
                "name": name,
                "label": f"LM Studio · {name}",
                "ctx": int(ctx),
                "loaded": loaded,
            })
        if out:
            return out
    except Exception:
        pass

    # Fallback: bare OpenAI-compatible listing has no ctx info at all.
    req = urllib.request.Request(base_v1 + "/models", method="GET")
    req.add_header("Authorization", f"Bearer {token or 'lm-studio'}")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return out
    for m in (data.get("data") or []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("id") or "").strip()
        if not name:
            continue
        out.append({"name": name, "label": f"LM Studio · {name}", "ctx": 32_768, "loaded": False})
    return out


def check_lmstudio_server(base_url: str = "", timeout: int = 4) -> bool:
    """Quick reachability check for an LM Studio server (``GET /v1/models``)."""
    raw = (base_url or "").strip() or _env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    base = _ensure_v1_base_url(raw)
    req = urllib.request.Request(base + "/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def get_llm(profile: str | None = None) -> Tuple[Any, ProfileName, str]:
    profile_name = normalize_profile(profile)
    cfg = _PROFILES[profile_name]
    model_name = str(cfg["model"])
    temperature = float(cfg["temperature"])
    max_tokens = int(cfg.get("max_tokens", 16384))

    if model_name.startswith("ollama/"):
        wire_model_name = model_name.split("/", 1)[1]
        settings = _resolve_ollama_settings(wire_model_name, temperature, max_tokens)
        if _HAS_CHAT_OLLAMA:
            llm = _build_ollama_chat_llm(wire_model_name, settings)
        else:
            llm = _build_ollama_openai_llm(wire_model_name, settings)
        return llm, profile_name, model_name

    if model_name.startswith("lmstudio/"):
        wire_model_name = model_name.split("/", 1)[1]
        settings = _resolve_lmstudio_settings(temperature, max_tokens)
        llm = _build_lmstudio_llm(wire_model_name, settings)
        return llm, profile_name, model_name

    base_url = env_pref("BASE_URL", "https://openrouter.ai/api/v1")
    api_key = _env("OPENROUTER_API_KEY", "")
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=3,
    )
    return llm, profile_name, model_name


def set_model(model_id: str) -> str:
    """Set model, persist, and rebuild profiles. Returns the model_id."""
    mid = (model_id or "").strip()
    if not mid:
        raise ValueError("пустой id модели")
    if mid in ("/exit", "/quit", "exit", "quit"):
        raise ValueError("некорректный id модели (похоже на команду выхода, а не на модель).")
    save_model_choice(mid)
    reload_profiles()
    return mid


# ─── OpenRouter API ────────────────────────────────────────────────

def fetch_openrouter_credits(api_key: str | None = None) -> Optional[Dict[str, Any]]:
    """Fetch account credits/usage from OpenRouter API.
    Returns dict with 'usage', 'limit', 'is_free_tier', 'rate_limit' or None on error.
    """
    key = (api_key or "").strip() or _env("OPENROUTER_API_KEY", "")
    if not key:
        return None

    url = "https://openrouter.ai/api/v1/auth/key"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", data)
    except Exception:
        return None


def fetch_openrouter_model_metadata(model_id: str, api_key: str = "") -> Optional[Dict[str, Any]]:
    """Fetch one model card from OpenRouter /models by id."""
    key = (api_key or "").strip() or _env("OPENROUTER_API_KEY", "")
    if not key or not model_id:
        return None
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", method="GET")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    rows = data.get("data", [])
    if not isinstance(rows, list):
        return None
    target = (model_id or "").strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        if rid.lower() == target:
            return row
    return None


def fetch_ollama_models(base_url: str = "", api_key: str = "") -> List[Dict[str, Any]]:
    """Fetch model list from Ollama /api/tags (local or remote)."""
    raw = (base_url or "").strip() or _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    req = urllib.request.Request(base + "/api/tags", method="GET")
    token = (api_key or "").strip() or _env("OLLAMA_API_KEY", "")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    out: List[Dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return out
    for m in (data.get("models") or []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        details = m.get("details") if isinstance(m.get("details"), dict) else {}
        out.append(
            {
                "name": name,
                "label": f"Ollama · {name}",
                "ctx": int(m.get("context_length") or details.get("context_length") or 32_768),
                "details": details,
            }
        )
    return out


def _ollama_http_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    base_url: str = "",
    api_key: str = "",
    timeout: int = 8,
) -> Optional[Dict[str, Any]]:
    raw = (base_url or "").strip() or _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + path
    body = None
    headers = {"Content-Type": "application/json"}
    token = (api_key or "").strip() or _env("OLLAMA_API_KEY", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, method=method.upper(), data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_out = resp.read().decode("utf-8", errors="ignore").strip()
            if not raw_out:
                return {}
            data = json.loads(raw_out)
            return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None


def fetch_ollama_running_models(base_url: str = "", api_key: str = "") -> List[str]:
    """Return currently loaded/running model names from Ollama /api/ps."""
    data = _ollama_http_json("GET", "/api/ps", base_url=base_url, api_key=api_key, timeout=6)
    if not isinstance(data, dict):
        return []
    rows = data.get("models") or []
    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name:
            out.append(name)
    return out


def unload_ollama_models(base_url: str = "", api_key: str = "") -> Dict[str, Any]:
    """Unload all currently running Ollama models (best-effort, no exceptions)."""
    running = fetch_ollama_running_models(base_url=base_url, api_key=api_key)
    unloaded = 0
    failed: List[str] = []
    for name in running:
        ok = _ollama_http_json(
            "POST",
            "/api/generate",
            payload={"model": name, "prompt": "", "stream": False, "keep_alive": 0},
            base_url=base_url,
            api_key=api_key,
            timeout=8,
        )
        if ok is None:
            failed.append(name)
        else:
            unloaded += 1
    return {"running": len(running), "unloaded": unloaded, "failed": failed[:12]}


def format_credits_info(data: Dict[str, Any]) -> str:
    """Format credits data into a readable string."""
    usage = data.get("usage", 0)
    limit = data.get("limit")
    is_free = data.get("is_free_tier", True)

    usage_daily = data.get("usage_daily", 0)
    usage_monthly = data.get("usage_monthly", 0)

    lines = []
    if limit is not None and limit > 0:
        remaining = max(0, limit - usage)
        lines.append(f"Баланс: ${remaining:.4f} (использовано ${usage:.4f} из ${limit:.4f})")
    else:
        lines.append(f"Использовано всего: ${usage:.4f}")
        if limit is None:
            lines.append("Лимит: неограничен")

    if usage_daily:
        lines.append(f"За сегодня: ${usage_daily:.4f}")
    if usage_monthly:
        lines.append(f"За месяц: ${usage_monthly:.4f}")

    lines.append(f"Тариф: {'бесплатный' if is_free else 'платный'}")

    return "\n".join(lines)
