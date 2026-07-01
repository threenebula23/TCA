MODULE: tests.test_creator_mode_local_tools

PURPOSE:
Python module `tests.test_creator_mode_local_tools`.

PUBLIC_API:
|name|description|
|---|---|
|test_creator_worker_recovers_local_tool_json_from_content||
|test_creator_worker_recovers_textual_tool_call_from_content||
|test_creator_provider_normalizes_plain_ollama_url_to_v1||

DEPENDENCIES:
- Agent.creator_mode
- Agent.creator_provider
- __future__
- langchain_core.messages
- langchain_core.tools
- types

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:

RISKS:
