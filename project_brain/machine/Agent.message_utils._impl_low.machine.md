MODULE: Agent.message_utils._impl_low

PURPOSE:
Message sanitization, compaction, and tool result processing utilities.

PUBLIC_API:
|name|description|
|---|---|
|is_retriable_bind_error|Check if the error is caused by an unsupported bind_tools parameter.|
|strip_think_tags|Remove every reasoning wrapper from visible output.

Covers XML-style tags (<think>, <thinking>, <thought>, <reasoning>,
<analysis>, <scratchpad>, <redacted_thinking>), Qwen pipe tags
(<|thinking|>…<|/thinking|>), bracketed markers ([THINKI…|
|extract_thought_segments|Split raw model output into (thought_segments, visible_body).

Recognised reasoning wrappers (all case-insensitive, content preserved in
emission order):
  - XML: <thought>, <think>, <thinking>, <reasoning>, <analysis>,
    <scratchpad>, <r…|
|coerce_assistant_content_to_text|Normalize provider-specific assistant content payloads to plain text.|
|extract_message_usage|Return {'input_tokens', 'output_tokens', 'total_tokens'} for an AIMessage.

Supports multiple provider shapes:
  - LangChain `usage_metadata` attribute (standard, emitted by ChatOllama/ChatOpenAI)
  - `response_metadata.usage` / `response_m…|
|extract_reasoning_from_response|Extract hidden reasoning/thought text from non-content response fields.|
|is_transient_error|Check if the error is a transient provider error that may resolve on retry.|
|sanitize_messages|Fix message history to prevent API errors.

Handles two classes of problems that cause 400 "No tool call found":
1. Orphaned ToolMessages — their AIMessage was lost (e.g. compaction/restore)
2. Dangling tool_calls — AIMessage has tool_calls…|
|sanitize_tool_call_name|Normalize provider-specific / hallucinated tool names (Qwen channel, namespaces, …).|
|register_known_tool_names|Record the list of real tool names available in the current session.|
|normalize_tool_call||

DEPENDENCIES:
- Agent.config
- Interface.visualization
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
