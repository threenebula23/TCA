MODULE: tests.test_package_imports

PURPOSE:
Регрессия: публичные точки входа после разбиения пакетов (>1000 строк).

PUBLIC_API:
|name|description|
|---|---|
|test_agent_message_utils_package||
|test_interface_code_editor_package||
|test_agent_deep_solver_package||
|test_interface_visualization_module||

DEPENDENCIES:
- Agent.deep_solver
- Agent.message_utils
- Interface
- Interface.panels.code_editor
- __future__

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:

RISKS:
