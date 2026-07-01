MODULE: Agent.stream_chat_mode

PURPOSE:
Thread-local slug режима чата на время ``agent_graph.stream`` (TUI / classic).

PUBLIC_API:
|name|description|
|---|---|
|set_stream_chat_mode|Выставить режим для текущего потока; ``None`` — сброс.|
|get_stream_chat_mode||

DEPENDENCIES:
- __future__
- threading

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_tui.py
- Agent/graph_runner.py
- tests/test_ollama_provider.py

RISKS:
