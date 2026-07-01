# Фоновые задачи и Deep Solver (обзор)

## Фоновый помощник

- Тулы: `start_background_task`, `get_background_result` (`Agent/tools/parallel_helper_tool.py`).
- Отдельный поток с собственным циклом LLM + tools; результат по токену.

## Суб-агенты (`spawn_subagent` / `get_subagent_result`)

Общая реализация — `Agent/subagent_runner.py`: `spawn()` стартует фоновый Creator-воркер (общий пул, **не более 3** одновременно, `threading.Semaphore`), `get_result(token, wait_seconds)` — опрос/ожидание. Доступны в Agent, Brainer, Research и Deep (диспетчеризуются отдельно от обычного bind_tools в цикле Deep, но через тот же раннер); **исключены в Ask** — суб-агент наследует полный набор тулов родителя, включая `run_command`/`edit_file`. Прогресс отображается и в чате, и деревом `main → children` в `ActiveAgentsPanel` (`TUIBridge.on_subagent_spawn`/`on_subagent_done`).

## Deep Solver

Долгий **локальный** автономный цикл: не обычный чат-граф; реализация в `Agent/deep_solver/` (см. `legacy_loop.py`). Чекпойнты, очередь сообщений пользователя во время прогона, суб-агенты Creator через `spawn_subagent` / `get_subagent_result` (общий `subagent_runner`, см. выше). По завершении итоговый отчёт автоматически пишется в `project_brain/agent/deep_report.md`.

**Детали режима, переменные окружения, чекпойнты:** [MODES/deep.md](MODES/deep.md).

## Связанные документы

- [MODES/README.md](MODES/README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
