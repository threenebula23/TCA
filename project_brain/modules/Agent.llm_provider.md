# Module: Agent.llm_provider

## Purpose

Провайдер LLM для Lorne: профили, сохранение конфигурации, OpenRouter.

---

## Responsibilities

- Провайдер LLM для Lorne: профили, сохранение конфигурации, OpenRouter.

---

## Public API

|name|description|
|---|---|
|get_ollama_model_quirks|Best-effort per-family defaults for an Ollama tag (e.g. ``qwen2.5:7b``).

Falls back to a generic "assume native tools, 32k ctx" entry for unknown
families. This does not replace explicit user overrides in
``ollama_model_settings`` — it onl…|
|fetch_ollama_model_show|Query Ollama ``/api/show`` for a model's real metadata (cached).

Used to clip a too-large ``num_ctx`` down to what the model actually
supports, instead of trusting a single hardcoded default for every tag.|
|get_ollama_model_max_ctx|Best-effort real context-length ceiling for a model, via ``/api/show``.

Returns ``None`` when unknown (offline server, unsupported field, etc.)
so callers can fall back to the family-quirk default instead.|
|get_model_capabilities|Return capability flags for a given model based on its provider prefix.

For ``ollama/<tag>`` models this refines the generic prefix-level flags
with the per-tag quirks registry, since native tool-calling support is
not uniform across local…|
|supports_parallel_tool_calls_param|Check if the model's provider supports the parallel_tool_calls binding parameter.|
|is_reasoning_model|Check if the model is a reasoning/thinking model that emits <think> blocks.|
|get_available_models|Curated models + user-added OpenRouter/Ollama models from UI prefs.|
|load_config||
|save_config||
|get_saved_model||
|save_model_choice||
|reload_profiles|Rebuild profiles after model change.|
|get_available_profiles||
|normalize_profile||
|fetch_lmstudio_models|Fetch model list with real context length from LM Studio's native API.

Tries ``GET /api/v0/models`` first (LM Studio-specific; reports
``max_context_length`` / ``loaded_context_length`` per model and whether a
model is actually loaded). Fa…|
|check_lmstudio_server|Quick reachability check for an LM Studio server (``GET /v1/models``).|
|get_llm||
|set_model|Set model, persist, and rebuild profiles. Returns the model_id.|
|fetch_openrouter_credits|Fetch account credits/usage from OpenRouter API.
Returns dict with 'usage', 'limit', 'is_free_tier', 'rate_limit' or None on error.|
|fetch_openrouter_model_metadata|Fetch one model card from OpenRouter /models by id.|
|fetch_ollama_models|Fetch model list from Ollama /api/tags (local or remote).|
|fetch_ollama_running_models|Return currently loaded/running model names from Ollama /api/ps.|
|unload_ollama_models|Unload all currently running Ollama models (best-effort, no exceptions).|
|format_credits_info|Format credits data into a readable string.|

---

## Dependencies

- `Agent.runtime_paths`
- `json`
- `langchain_ollama`
- `langchain_openai`
- `os`
- `pathlib`
- `time`
- `typing`
- `urllib.error`
- `urllib.request`

---

## Used By

- `Agent/agent/_impl_prepare.py`
- `Agent/command_router/_main.py`
- `Agent/command_router/_mixin_handlers.py`
- `Agent/creator_provider.py`
- `Agent/deep_solver/_impl_a.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/planner.py`
- `Agent/tool_registry.py`
- `Agent/tools/parallel_helper_tool.py`
- `Interface/panels/ai_chat/_mixin_events.py`
- `Interface/panels/file_explorer.py`
- `Interface/visualization.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/llm_provider.py`

---

## Entry Points


---

## API / route hints

