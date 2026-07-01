# Module: Agent.stream_chat_mode

## Purpose

Thread-local slug режима чата на время ``agent_graph.stream`` (TUI / classic).

---

## Responsibilities

- Thread-local slug режима чата на время ``agent_graph.stream`` (TUI / classic).

---

## Public API

|name|description|
|---|---|
|set_stream_chat_mode|Выставить режим для текущего потока; ``None`` — сброс.|
|get_stream_chat_mode||

---

## Dependencies

- `__future__`
- `threading`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_tui.py`
- `Agent/graph_runner.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/stream_chat_mode.py`

---

## Entry Points


---

## API / route hints

