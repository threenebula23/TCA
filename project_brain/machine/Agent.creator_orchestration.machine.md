MODULE: Agent.creator_orchestration

PURPOSE:
Режимы оркестрации Creator Mode (мультиагентность: параллель, конвейер, супервайзер, иерархия).

PUBLIC_API:
|name|description|
|---|---|
|normalize_orchestration||
|worker_roles_for_count|Return role names for n workers (used for system prompt and tree display).|
|role_hint|Return full role prompt markdown (loaded from Agent/roles/).|
|format_worker_mode_section|Блок для SystemMessage после SYSTEM_PROMPT и project_context.|
|build_worker_user_content||
|synthesize_supervisor_report|Один проход тяжёлой модели: единый отчёт по всем воркерам.|

DEPENDENCIES:
- __future__
- langchain_core.messages
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/creator_mode.py
- tests/test_ollama_provider.py

RISKS:
