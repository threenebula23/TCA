MODULE: Agent.creator_provider

PURPOSE:
Creator Provider — маршрутизация между локальной и тяжёлой моделью.

PUBLIC_API:
|name|description|
|---|---|
|get_creator_config|Загрузить конфигурацию creator mode.

UI-level настройки (orchestration / max_workers) имеют приоритет над project
config, чтобы пользователь мог менять их из экрана Settings → Agents без
перезапуска приложения. Это работает одинаково для л…|
|save_creator_config|Сохранить настройки creator mode.|
|get_local_llm|Создаёт LLM для локального сервера.

Автоматически выбирает транспорт:
- native Ollama через ChatOllama (`/api/chat`), когда URL похож на Ollama
  daemon (порт 11434) и `langchain-ollama` доступен — это даёт правильную
  обработку tool-call…|
|get_heavy_llm|Создаёт ChatOpenAI через OpenRouter (тяжёлая модель).

Returns:
    (ChatOpenAI, model_name)|
|check_local_server|Проверить доступность локального сервера (включая аутентификацию).

Пробует разные варианты API в порядке популярности:
  • OpenAI-совместимые: ``/v1/models``, ``/models``.
  • Ollama native: ``/api/tags``, ``/api/version``.
  • OpenWebUI: …|
|classify_task_complexity|Определить сложность задачи.

Args:
    task_text: Текст задачи
    plan_steps: Количество шагов в плане (если известно)

Returns:
    "simple" или "complex"|
|route_to_model|Маршрутизировать задачу на подходящую модель.

Returns:
    (llm, model_name, model_type) — model_type: "local" или "heavy"|

DEPENDENCIES:
- Agent.llm_provider
- Interface.ui_prefs
- __future__
- langchain_openai
- llm_provider
- os
- re
- typing
- urllib.error
- urllib.request

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/creator_mode.py
- Agent/deep_solver/_impl_a.py
- Agent/deep_solver/legacy_loop.py
- tests/test_creator_mode_local_tools.py
- tests/test_ollama_provider.py

RISKS:
