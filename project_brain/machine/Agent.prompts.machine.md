MODULE: Agent.prompts

PURPOSE:
Per-mode system prompt fragments for Lorne v1.0.

PUBLIC_API:
|name|description|
|---|---|
|mode_prompt_addon|Return system prompt fragment for *mode* slug, or empty string.|

DEPENDENCIES:
- __future__
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- tests/test_ollama_provider.py

RISKS:
