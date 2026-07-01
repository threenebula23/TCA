MODULE: Interface.input_widget

PURPOSE:
Advanced input widget with @ file autocomplete and /command completion.

PUBLIC_API:
|name|description|
|---|---|
|add_to_history||
|get_project_files||
|invalidate_file_cache||
|get_user_input_advanced|Prompt with @ file completion, /command completion, and history.|
|get_file_suggestions|Get file suggestions for @ trigger in Textual Input.|
|get_command_suggestions|Get command suggestions for / trigger in Textual Input.|

DEPENDENCIES:
- Interface.ui_prefs
- __future__
- os
- pathlib
- prompt_toolkit
- prompt_toolkit.completion
- prompt_toolkit.history
- prompt_toolkit.styles
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
