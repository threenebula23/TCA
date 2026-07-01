# Module: tests.test_message_utils_tools

## Purpose

Tool-call recovery for Ollama-style OpenAI responses.

---

## Responsibilities

- Tool-call recovery for Ollama-style OpenAI responses.

---

## Public API

|name|description|
|---|---|
|test_coalesce_prefers_normal_tool_calls||
|test_coalesce_recovers_dict_args_from_invalid_tool_calls||
|test_coalesce_recovers_json_string_args_from_invalid||
|test_coalesce_handles_openai_function_shape_in_primary||
|test_coalesce_applies_alias_and_reasoning_repairs||
|test_coalesce_recovers_from_additional_kwargs_tool_calls||
|test_normalize_alias_create_file_to_write_file||
|test_sanitize_qwen_channel_prefix_reasoning_tool||
|test_sanitize_google_search_alias||
|test_repair_thought_glued_with_ask_user_json||
|test_normalize_mangled_reasoning_tool_name||
|test_normalize_unwraps_assistant_meta_tool_to_ask_user||
|test_normalize_unwraps_assistant_meta_tool_to_reasoning_tool||
|test_extract_textual_tool_call_from_assistant_kwargs_text||
|test_extract_textual_tool_call_from_reasoning_kwargs_text||
|test_extract_thought_segments_qwen_and_thought||
|test_extract_reasoning_from_response_uses_metadata_and_additional_kwargs||
|test_coerce_assistant_content_to_text_handles_list_fragments||
|test_extract_reasoning_from_response_reads_nested_ollama_message||
|test_extract_thought_segments_handles_dangling_think_tag||
|test_extract_structured_tool_calls_plain_name_arguments_shape||
|test_coalesce_recovers_nested_response_metadata_tool_calls||
|test_summarize_tool_like_final_answer_for_file_write_payload||
|test_summarize_tool_like_final_answer_for_recorded_payload||
|test_normalize_alias_create_code_file_to_code_file_tool||
|test_textual_tool_calls_ignore_inline_python_when_tools_registered|Code lines like `print(x)` / `if b=0` must NOT become tool calls.|
|test_textual_tool_calls_accept_registered_tool_line|A line that really is a tool call is still recognized.|
|test_textual_tool_calls_reject_bullet_code_lines|Bullet-prefixed code noise from Ollama must not be promoted to tools.|
|test_strip_think_tags_handles_harmony_channels|gpt-oss / Harmony `<|channel|>analysis<|message|>` becomes invisible; `final` survives.|
|test_extract_thought_segments_splits_harmony_channels||
|test_extract_thought_segments_handles_harmony_with_leading_plain_text||
|test_extract_message_usage_reads_langchain_usage_metadata||
|test_extract_message_usage_reads_openai_response_metadata||
|test_extract_message_usage_reads_native_ollama_counters|Native Ollama exposes prompt_eval_count / eval_count at meta top level.|
|test_extract_message_usage_returns_zero_when_missing||
|test_compact_conversation_shrinks_old_tool_results|Old big tool results get excerpted rather than kept whole.|
|test_compact_conversation_keeps_recent_unchanged||
|test_strip_think_tags_handles_all_xml_variants||
|test_extract_thought_segments_xml_variants_preserves_order||
|test_extract_thought_segments_qwen_pipe_tag|Qwen ChatML-style pipe tokens <|thinking|>…<|/thinking|>.|
|test_extract_thought_segments_bracket_markers|Some finetunes emit [THINKING]…[/THINKING] instead of XML.|
|test_extract_thought_segments_bracket_reasoning_variant||
|test_strip_think_tags_removes_dangling_bracket_marker||
|test_extract_thought_segments_dangling_reasoning_tag||
|test_extract_reasoning_from_response_anthropic_thinking_block|Claude content block: {'type': 'thinking', 'thinking': '...', 'signature': '...'}|
|test_extract_reasoning_from_response_deepseek_reasoning_content|DeepSeek-R1 on OpenAI-compat: reasoning_content sits in additional_kwargs.|
|test_extract_reasoning_from_response_openrouter_reasoning_string|OpenRouter surfaces reasoning as a top-level string on the message.|
|test_extract_reasoning_from_response_openai_responses_summary|OpenAI Responses API: reasoning.summary[i].text.|
|test_extract_reasoning_from_response_openai_reasoning_summary_string|Some deployments flatten reasoning_summary to a plain string.|
|test_extract_reasoning_from_response_gemini_thought_flag|Gemini marks reasoning parts with {'thought': True, 'text': '...'}|
|test_extract_reasoning_from_response_cohere_tool_plan|Cohere Command emits tool_plan in additional_kwargs/response_metadata.|
|test_extract_reasoning_from_response_avoids_nontext_values|Boolean/numeric noise must not explode the parser.|
|test_extract_thought_segments_mixed_xml_and_harmony||
|test_compact_conversation_noop_for_short_history||
|test_build_aimessage_from_ollama_tool_parse_error_recovers_write_file|Ollama/Go rejects JSON where a string has `\` before a backtick (``` in markdown).|
|test_build_aimessage_from_ollama_tool_parse_error_recovers_run_command_equals_syntax|Models sometimes emit ``timeout_seconds=120`` instead of JSON ``"timeout_seconds": 120``.|

---

## Dependencies

- `Agent.message_utils`
- `__future__`
- `langchain_core.messages`
- `types`

---

## Used By


---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `tests/test_message_utils_tools.py`

---

## Entry Points


---

## API / route hints

