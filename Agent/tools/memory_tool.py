"""Persistent in-session key-value memory for the agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


def _mem_path() -> Path:
    return Path(".lorne") / "session_memory.json"


def _load(namespace: str = "default") -> dict:
    p = _mem_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:
        data = {}
    return data.get(namespace, {})


def _save_ns(namespace: str, ns_data: dict) -> None:
    p = _mem_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:
        data = {}
    data[namespace] = ns_data
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@tool
def structured_memory(action: str, key: str = "", value: str = "", namespace: str = "default") -> dict:
    """Сессионная KV-память: set/get/list/delete/clear.

    action: set | get | list | delete | clear
    key: ключ записи (для set/get/delete)
    value: значение (для set)
    namespace: пространство имён (по умолчанию 'default'; воркеры Creator используют worker_id)

    Храни сюда: архитектурные факты, пути к ключевым файлам, договорённости,
    промежуточные результаты. Память живёт в .lorne/session_memory.json.
    """
    ns = _load(namespace)
    act = (action or "").strip().lower()

    if act == "set":
        if not key:
            return {"error": "key required for set"}
        ns[key] = value
        _save_ns(namespace, ns)
        return {"ok": True, "key": key, "namespace": namespace}

    elif act == "get":
        if not key:
            return {"error": "key required for get"}
        v = ns.get(key)
        if v is None:
            return {"found": False, "key": key}
        return {"found": True, "key": key, "value": v}

    elif act == "list":
        return {"namespace": namespace, "entries": ns, "count": len(ns)}

    elif act == "delete":
        if not key:
            return {"error": "key required for delete"}
        existed = key in ns
        ns.pop(key, None)
        _save_ns(namespace, ns)
        return {"ok": True, "deleted": existed}

    elif act == "clear":
        _save_ns(namespace, {})
        return {"ok": True, "namespace": namespace, "cleared": True}

    return {"error": f"Unknown action '{action}'. Valid: set|get|list|delete|clear"}
