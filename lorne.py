#!/usr/bin/env python3
"""
Lorne v1.0 — Vi-like Terminal IDE.

Точка входа: этот файл (после ``install.sh`` / ``install.bat`` в PATH — команда ``lorne``).

Примеры::

    lorne
    lorne /path/to/project
    lorne env=<OPENROUTER_API_KEY>
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_env_key = ""
_filtered_argv = [sys.argv[0]]
for _arg in sys.argv[1:]:
    if _arg.startswith("env="):
        _env_key = _arg[4:]
    else:
        _filtered_argv.append(_arg)
sys.argv = _filtered_argv

_env_agent = _REPO_ROOT / "Agent" / ".env"
if _env_key:
    os.environ["OPENROUTER_API_KEY"] = _env_key
    _env_agent.parent.mkdir(parents=True, exist_ok=True)
    _env_agent.write_text(f"OPENROUTER_API_KEY={_env_key}\n", encoding="utf-8")
    print("\n  Ключ сохранён в Agent/.env — в следующий раз достаточно: lorne\n")
_env_root = _REPO_ROOT / ".env"
if _env_agent.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_agent)
elif _env_root.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_root)


def main():
    # Remove legacy --tui flag (now default and only mode)
    if "--tui" in sys.argv:
        sys.argv.remove("--tui")

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        target = Path(sys.argv[1]).resolve()
        if target.is_dir():
            os.chdir(target)
        else:
            print(f"Директория не найдена: {target}")
            sys.exit(1)

    from Agent.agent import run_tui_mode
    run_tui_mode()


if __name__ == "__main__":
    main()
