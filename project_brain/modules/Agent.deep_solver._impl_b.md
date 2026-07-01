# Module: Agent.deep_solver._impl_b

## Purpose

Вторая часть Deep Solver: основной цикл в :mod:`Agent.deep_solver.legacy_loop`.

---

## Responsibilities

- Вторая часть Deep Solver: основной цикл в :mod:`Agent.deep_solver.legacy_loop`.

---

## Public API

|name|description|
|---|---|
|run_deep_solver||
|apply_checkpoint_action|Rollback / continue from a Deep checkpoint (uses shared index in ``_impl_a``).|

---

## Dependencies

- `Agent.checkpoint`
- `__future__`
- `checkpoint`
- `importlib.util`
- `pathlib`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_tui.py`
- `tests/test_deep_solver.py`
- `tests/test_ollama_provider.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/deep_solver/_impl_b.py`

---

## Entry Points


---

## API / route hints

