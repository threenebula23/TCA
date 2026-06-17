"""Environment information tool."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


@tool
def env_info(check_packages: Optional[List[str]] = None,
             check_commands: Optional[List[str]] = None) -> dict:
    """Информация об окружении: Python, пакеты, команды, ОС.

    check_packages: ['django','numpy'] — проверить версии конкретных пакетов.
    check_commands: ['git','node','npm'] — проверить наличие команд в PATH.
    Вызывай в начале сессии или перед установкой зависимостей.
    Возвращает {python_version, os, cwd, checked_packages, checked_commands}.
    """
    result: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "cwd": os.getcwd(),
        "env_vars_count": len(os.environ),
    }

    pkg_results: Dict[str, Optional[str]] = {}
    for pkg in (check_packages or [])[:20]:
        spec = importlib.util.find_spec(pkg.replace("-", "_"))
        if spec is not None:
            try:
                import importlib.metadata
                version = importlib.metadata.version(pkg)
                pkg_results[pkg] = version
            except Exception:
                pkg_results[pkg] = "installed (version unknown)"
        else:
            pkg_results[pkg] = None
    if pkg_results:
        result["checked_packages"] = pkg_results

    cmd_results: Dict[str, Optional[str]] = {}
    for cmd in (check_commands or [])[:20]:
        path = shutil.which(cmd)
        cmd_results[cmd] = path
    if cmd_results:
        result["checked_commands"] = cmd_results

    return result
