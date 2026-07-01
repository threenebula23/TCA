# Module: Interface.panels.diff_block

## Purpose

Side-by-side code diff shown in the chat stream after a file-editing tool.

---

## Responsibilities

- Side-by-side code diff shown in the chat stream after a file-editing tool.

---

## Public API

|name|description|
|---|---|
|diff_stats|Count the number of added / removed lines between two texts.|
|read_before_after_texts|Best-effort retrieval of the *before* and *after* content of a file.

Uses the versioning store for the pre-edit snapshot and the on-disk
content for the post-edit view. Either side may be empty if unavailable.|

---

## Dependencies

- `Agent.versioning`
- `Interface.themes`
- `Interface.ui_prefs`
- `__future__`
- `difflib`
- `pathlib`
- `rich.text`
- `textual.app`
- `textual.containers`
- `textual.widgets`
- `typing`

---

## Used By

- `Interface/panels/ai_chat/__init__.py`
- `Interface/panels/ai_chat/_accent_dialog.py`
- `Interface/panels/ai_chat/_blocks.py`
- `Interface/panels/ai_chat/_messages.py`
- `Interface/panels/ai_chat/_mixin_events.py`
- `Interface/panels/ai_chat/_mixin_setup.py`
- `Interface/panels/ai_chat/_mixin_stream.py`
- `tests/test_branding.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Interface/panels/diff_block.py`

---

## Entry Points


---

## API / route hints

