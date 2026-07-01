MODULE: Agent.deep_solver._impl_b

PURPOSE:
Вторая часть Deep Solver: основной цикл в :mod:`Agent.deep_solver.legacy_loop`.

PUBLIC_API:
|name|description|
|---|---|
|run_deep_solver||
|apply_checkpoint_action|Rollback / continue from a Deep checkpoint (uses shared index in ``_impl_a``).|

DEPENDENCIES:
- Agent.checkpoint
- __future__
- checkpoint
- importlib.util
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_tui.py
- tests/test_deep_solver.py
- tests/test_ollama_provider.py
- tests/test_package_imports.py

RISKS:
