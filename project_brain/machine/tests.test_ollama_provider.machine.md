MODULE: tests.test_ollama_provider

PURPOSE:
Ollama-specific branches of llm_provider: settings layering, URL normalization,

PUBLIC_API:
|name|description|
|---|---|
|test_ensure_native_base_url_strips_v1_suffix||
|test_ensure_v1_base_url_adds_suffix_once||
|test_get_model_capabilities_has_explicit_ollama_entry||
|test_supports_parallel_tool_calls_param_false_for_ollama||
|test_resolve_ollama_settings_uses_defaults_when_no_prefs||
|test_resolve_ollama_settings_preset_layer_applied||
|test_resolve_ollama_settings_per_model_overrides_preset||
|test_env_vars_win_over_ui_prefs_for_connection||
|test_openai_fallback_builds_chat_openai_with_supported_params_only||
|test_get_llm_returns_ollama_llm_for_ollama_prefix||
|test_get_llm_uses_chat_ollama_when_available|If langchain-ollama is installed, we must use it — ChatOllama forwards
all native options (num_ctx/top_k/repeat_penalty) which ChatOpenAI cannot.|

DEPENDENCIES:
- Agent
- __future__
- json
- os
- pathlib
- pytest

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:

RISKS:
