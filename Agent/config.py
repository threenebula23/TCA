"""Центральные настройки Lorne v0.99.

Импортируй отсюда вместо хардкода чисел:

    from Agent.config import MAX_LLM_RETRIES, WEB_SEARCH_TTL
"""
from __future__ import annotations

MAX_LLM_RETRIES: int = 2

WEB_SEARCH_TTL: int = 300
WEB_FETCH_TTL: int = 600

DEEP_CHECKPOINT_INTERVAL: int = 5
DEEP_MAX_ITERATIONS: int = 40

CALENDAR_WEEKS: int = 17

MULTI_READ_MAX_FILES: int = 8
