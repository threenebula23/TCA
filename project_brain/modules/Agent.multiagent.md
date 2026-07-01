# Module: Agent.multiagent

## Purpose

Простая реализация multiagent: логические "под-агенты" (чаты) поверх одной модели.

---

## Responsibilities

- Простая реализация multiagent: логические "под-агенты" (чаты) поверх одной модели.

---

## Public API

|name|description|
|---|---|
|create_agent|Создаёт или переинициализирует под-агента с заданным id.|
|list_agents|Возвращает список известных под-агентов.|
|set_current_agent|Делает указанного под-агента текущим (если он существует).|
|get_current_agent|Возвращает id текущего под-агента.|
|get_agent_count|Возвращает текущее число логических под-агентов.|

---

## Dependencies

- `typing`

---

## Used By

- `Agent/command_router/_mixin_handlers.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/multiagent.py`

---

## Entry Points


---

## API / route hints

