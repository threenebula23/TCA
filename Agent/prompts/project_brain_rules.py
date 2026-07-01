"""Project Brain + RAG rules (workspace brain, not the IDE package).

J2/J3 (экономия контекста): раньше этот блок нёс полную инструкцию по каждому
режиму сразу в базовом system prompt, даже когда сессия ни разу не тронет
``project_brain_tool`` (например Ask). Здесь остаётся только компактное
ядро — куда можно/нельзя писать и общий приоритет brain-first RAG; детальный
workflow (когда именно писать, критерии) теперь только в ``brainer.md`` addon,
инжектится лишь при переключении в Brainer.
"""

PROJECT_BRAIN_SYSTEM_SECTION = """
=== PROJECT BRAIN (внешняя память) ===
Каталог ``project_brain/`` — Markdown, часть **пересобирается** сканером
(``project_brain_tool`` ``refresh``/``scan``), часть пишет только модель.

1. Архитектура/модули/связи — сначала ``rag_search`` (brain выше кода), затем код. Не выдумывай без опоры на brain или проверенный код.
2. Запись моделью — ``action=write_brain``, ``brain_rel_path`` + ``content``: только ``agent/…/*.md``, корневые ``*_notes.md``/``*_supplement.md`` или ``agent_architecture.md`` (``action=write_architecture``). **Не писать** в ``overview.md``/``architecture.md``/``glossary.md``/``tools.md``/``flows.md``/``modules/``/``machine/``/``services/``/``agents/`` — их даёт сканер.
3. ``reindex`` — быстро перечитать brain в RAG с диска; ``refresh``/``scan`` — полный пересбор из кода (дороже, только после структурных изменений).
"""
