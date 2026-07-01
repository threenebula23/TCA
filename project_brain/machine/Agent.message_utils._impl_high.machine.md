MODULE: Agent.message_utils._impl_high

PURPOSE:
Вторая половина утилит сообщений (после :func:`~Agent.message_utils._impl_low.normalize_tool_call`).

PUBLIC_API:
|name|description|
|---|---|
|tool_repetition_loop_nudge|When the model repeats the same tool+args many times, return an anti-loop hint (ephemeral).|
|extract_textual_tool_calls|Recover tool calls from plain-text pseudo-calls emitted by local models.|
|extract_structured_tool_calls|Recover JSON-structured tool calls from assistant text payload.|
|extract_inline_write_file_args|Best-effort recovery of inline write_file payloads from plain text.|
|summarize_tool_like_final_answer|Convert raw tool-result JSON into a concise user-facing sentence.|
|coalesce_lc_response_tool_calls|Return tool_calls from a LangChain chat-model response, with Ollama recovery.

Ollama's OpenAI-compatible endpoint may return ``function.arguments`` as a JSON
**object** instead of a string. LangChain's OpenAI converter then fails
``json.lo…|
|reconstruct_broken_content|Fix broken tool call args where the LLM failed to JSON-escape multi-line code.|
|build_aimessage_from_ollama_tool_parse_error|If Ollama could not parse tool call JSON, rebuild :class:`AIMessage` for one tool.|
|safe_chat_invoke_with_tool_recovery|``llm.invoke``; on Ollama tool-call parse errors, recover a synthetic ``AIMessage``.|
|truncate_result|Truncate tool result content to save context tokens.|
|annotate_errors|Add explicit error prefix to tool results so the model cannot ignore failures.|
|compact_conversation|Summarize old messages to free up context window.

Default `keep_last=8` (was 10) because tool results are the main context
hog — holding fewer full-fidelity turns while summarising the rest with a
tiny excerpt of every old tool result scal…|

DEPENDENCIES:
- Interface.visualization
- __future__
- _impl_low
- ast
- json
- json_repair
- langchain_core.messages
- re
- typing
- uuid

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- Agent/background_agent_runner.py
- Agent/checkpoint/__init__.py
- Agent/command_router/_main.py
- Agent/creator_mode.py
- Agent/deep_solver/_impl_a.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Interface/panels/ai_chat/_helpers.py
- Interface/panels/ai_chat/_mixin_stream.py
- tests/test_message_utils_tools.py
- tests/test_ollama_provider.py
- tests/test_package_imports.py

RISKS:
