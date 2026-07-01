"""
Task planner for Lorne — generates structured plans for complex tasks.
Only creates detailed plans for non-trivial tasks that benefit from planning.
"""
from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

try:
    from .llm_provider import get_llm
except ImportError:
    from Agent.llm_provider import get_llm


class _PlanSteps(BaseModel):
    """Schema for a planner response: a non-empty list of step strings."""
    steps: List[str]


def _extract_json_array(raw: str) -> Optional[list]:
    """Pull a JSON array out of raw LLM text, tolerating markdown fences."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def _validate_plan_steps(raw: str, max_steps: int) -> Optional[List[str]]:
    """Parse + validate a planner response against ``_PlanSteps``.

    Returns ``None`` (rather than a silently-degraded guess) when the model's
    output isn't a usable JSON array of strings, so the caller can retry once
    before falling back to a generic plan.
    """
    arr = _extract_json_array(raw)
    if arr is None:
        return None
    try:
        validated = _PlanSteps(steps=[str(x) for x in arr])
    except ValidationError:
        return None
    steps = [s.strip() for s in validated.steps if s.strip()]
    return steps[:max_steps] if steps else None


PLANNER_SYSTEM_PROMPT = """You are a task planner for a coding assistant.
Given a user request, return a concise, actionable list of steps (3-10) needed to complete the task.

Rules:
1. Each step should be a single, clear action (e.g., "Create file X with Y functionality")
2. Steps should be in execution order — dependencies first
3. Include testing/verification steps where appropriate
4. Be specific — mention file names, function names, technologies
5. Don't include meta-steps like "understand the request" — jump straight to action

Respond with ONLY a JSON array of strings, no other text. Example:
["Step 1: Create src/auth/handler.py with JWT token validation", "Step 2: Add login endpoint to src/routes/auth.py", "Step 3: Create tests in tests/test_auth.py", "Step 4: Run tests to verify", "Step 5: Update README.md with auth documentation"]
"""


CREATOR_PLANNER_SYSTEM_PROMPT = """You are a planner for a multi-worker coding orchestrator (Creator mode).
Decompose the user task into 2–6 LARGE work packages (not micro-steps).

Rules:
1. Each step is a coherent slice of work a single worker can finish (e.g. entire feature area, full module, or end-to-end slice with tests).
2. Prefer fewer, bigger steps over many tiny ones (no "add one line to README" unless essential).
3. Order by dependencies first.
4. Mention concrete paths/modules when obvious.
5. Return ONLY a JSON array of strings, no markdown fences.

Example:
["Implement core calculator module in src/calculator.py with operations and validation", "Add CLI in src/main.py and wire to calculator", "Add tests/test_calculator.py and run pytest", "Document usage in README.md"]
"""


_RETRY_FORMAT_NUDGE = (
    "Your previous response was not a valid JSON array of strings. "
    "Reply again with ONLY a JSON array of strings, no markdown fences, no extra text."
)


def build_plan(user_task: str) -> List[str]:
    """Call the LLM in planning mode and return a list of steps.

    Validates the response against a strict JSON-array-of-strings schema and
    retries once with a corrective nudge before falling back to a naive
    line-split (and finally a generic hardcoded plan) — this avoids silently
    accepting whatever the model wrote when it ignores the format.
    """
    if not user_task.strip():
        return []

    llm, _, _ = get_llm("fast")

    messages: List[object] = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_task),
    ]

    raw = ""
    for attempt in range(2):
        try:
            resp = llm.invoke(messages)
        except Exception:
            return _fallback_plan(user_task)
        raw = (resp.content or "").strip()
        steps = _validate_plan_steps(raw, max_steps=12)
        if steps:
            return steps
        if attempt == 0:
            messages.append(AIMessage(content=raw))
            messages.append(HumanMessage(content=_RETRY_FORMAT_NUDGE))

    # Last resort: naive line-split, then the generic hardcoded plan.
    lines = [ln.strip("-•* ").strip() for ln in raw.splitlines()]
    steps = [ln for ln in lines if ln and len(ln) > 5]
    if steps:
        return steps[:12]

    return _fallback_plan(user_task)


def build_creator_plan(user_task: str) -> List[str]:
    """Planner tuned for Creator: fewer, larger parallelizable subtasks.

    Same validate-then-retry-once strategy as ``build_plan`` (see there).
    """
    if not user_task.strip():
        return []

    llm, _, _ = get_llm("fast")
    messages: List[object] = [
        SystemMessage(content=CREATOR_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_task),
    ]
    for attempt in range(2):
        try:
            resp = llm.invoke(messages)
        except Exception:
            return _fallback_creator_plan(user_task)
        raw = (resp.content or "").strip()
        steps = _validate_plan_steps(raw, max_steps=8)
        if steps:
            return steps
        if attempt == 0:
            messages.append(AIMessage(content=raw))
            messages.append(HumanMessage(content=_RETRY_FORMAT_NUDGE))

    return _fallback_creator_plan(user_task)


def _fallback_creator_plan(user_task: str) -> List[str]:
    return [
        f"Deliver the requested feature end-to-end: {user_task[:200]}",
        "Verify with tests or manual checks and fix regressions",
    ]


def _fallback_plan(user_task: str) -> List[str]:
    """Generate a generic plan when LLM planning fails."""
    return [
        f"Step 1: Analyze the request: {user_task[:100]}",
        "Step 2: Read relevant files and understand the codebase",
        "Step 3: Implement the required changes",
        "Step 4: Verify the changes work correctly",
    ]
