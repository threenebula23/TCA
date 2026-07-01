MODULE: tests.test_deep_solver

PURPOSE:
Smoke tests for the Deep Solver plumbing — not the long run itself.

PUBLIC_API:
|name|description|
|---|---|
|test_deep_extra_tools_have_expected_names||
|test_filter_tools_drops_human_interactive_tools||
|test_extract_facts_read_file||
|test_extract_facts_write_file||
|test_render_tool_result_truncates_long_strings||
|test_render_tool_result_dumps_dict_json||
|test_checkpoint_registry_roundtrip||
|test_format_elapsed_branches||
|test_deep_state_singleton_queues_messages||
|test_deep_state_checkpoint_timing_monotonic||
|test_compact_with_head_lock_preserves_head||

DEPENDENCIES:
- Agent.deep_solver
- __future__
- langchain_core.messages
- pytest

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:

RISKS:
