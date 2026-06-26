"""Interactive Kanban clarifier — human-in-loop spec sharpening.

This is the Luna-style companion to ``kanban_specify``:

* ``clarify_next`` asks exactly one high-leverage question, or decides the
  task is ready and drafts a clarified spec.
* ``record_answer`` stores the human answer as a task comment.
* ``finalize_spec`` writes a clarified spec into the task body while keeping
  the task in ``triage`` so the existing Hermes Specify/Decompose passes can
  polish and/or fan it out.

All conversation state lives in task comments with stable markers. That keeps
this feature schema-free, auditable, and visible to later workers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from utils import env_int, is_truthy_value

logger = logging.getLogger(__name__)

QUESTION_PREFIX = "CLARIFY_QUESTION"
ANSWER_PREFIX = "CLARIFY_ANSWER"
READY_PREFIX = "CLARIFY_READY"

HERMES_KANBAN_CLARIFY_MAX_TOKENS = max(
    1200,
    env_int("HERMES_KANBAN_CLARIFY_MAX_TOKENS", 5000),
)

_SYSTEM_PROMPT = """You are the interactive clarifier for a Hermes Kanban triage task.

You run while a human operator is present. Your job is to sharpen a vague task
BEFORE any headless worker starts. Decide whether one more high-leverage human
question is needed, or whether the task is ready enough to draft a spec.

Return STRICT JSON only.

For another question:
{
  "state": "question",
  "question": "Ask exactly one concise, high-leverage question",
  "choices": ["2-4 concrete choices; prefer hypotheses over generic options"],
  "rationale": "why this answer matters for safe execution"
}

For ready:
{
  "state": "ready",
  "spec_markdown": "## Clarified Spec\n...",
  "rationale": "why enough is known"
}

Rules:
- Ask at most ONE question.
- Prefer 2-4 concrete choices. Include an "Other / custom" style option only
  when the choice space is genuinely open.
- Do not ask trivia. Ask only for decisions that materially affect scope,
  safety, route, acceptance criteria, or constraints.
- If the current task and transcript are enough, return ready instead of asking
  another question.
- Never invent repo paths, credentials, PR numbers, test results, Slack/Jira
  facts, or decisions not present in the task/transcript.
- Treat human answers in the clarify transcript as authoritative.
- The ready spec must include: Goal, Background, Scope Included, Scope Excluded,
  Acceptance Criteria, Constraints, Open Decisions, Suggested Execution Route.
- Unknowns should remain explicit Open Decisions, not fabricated answers.
- No preamble, no markdown fences around the JSON.
"""

_USER_TEMPLATE = """Task id: {task_id}
Current title: {title}
Current status: {status}
Current body:
{body}

Clarify transcript:
{transcript}

Instruction: {instruction}
"""

FALLBACK_QUESTIONS = [
    {
        "question": "What should the final deliverable be?",
        "choices": ["Code change + verification", "Design/spec document", "Research summary", "Manual/ops action"],
        "rationale": "The execution route and completion criteria depend on the deliverable.",
    },
    {
        "question": "How should success be verified?",
        "choices": ["Automated tests/build", "Manual UI/behavior check", "Reviewer approval", "Documented checklist"],
        "rationale": "The worker needs a concrete stop condition.",
    },
    {
        "question": "What scope boundary matters most?",
        "choices": ["No schema changes", "No broad UI redesign", "No production/ops side effects", "No unrelated refactors"],
        "rationale": "Explicit boundaries prevent over-implementation.",
    },
]


@dataclass
class ClarifyOutcome:
    task_id: str
    ok: bool
    state: str
    reason: str = ""
    question: Optional[str] = None
    choices: list[str] = field(default_factory=list)
    rationale: Optional[str] = None
    spec_markdown: Optional[str] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_blob(raw: str) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        parsed = json.loads(stripped[first : last + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _profile_author() -> str:
    return os.environ.get("HERMES_PROFILE") or os.environ.get("USER") or "clarifier"


def clarify_transcript(comments: list[kb.Comment]) -> str:
    rows: list[str] = []
    for c in comments:
        body = c.body or ""
        if body.startswith((QUESTION_PREFIX, ANSWER_PREFIX, READY_PREFIX)):
            rows.append(f"--- {c.author} @ {c.created_at} ---\n{body}")
    return "\n\n".join(rows) or "(none yet)"


def latest_question(comments: list[kb.Comment]) -> Optional[dict[str, Any]]:
    for c in reversed(comments):
        body = c.body or ""
        if not body.startswith(QUESTION_PREFIX):
            continue
        payload = body[len(QUESTION_PREFIX):].strip()
        if payload.startswith(":"):
            payload = payload[1:].strip()
        parsed = _extract_json_blob(payload)
        if parsed:
            return parsed
        return {"question": payload, "choices": [], "rationale": None}
    return None


def _normalize(parsed: dict[str, Any], task_id: str) -> ClarifyOutcome:
    state = str(parsed.get("state") or "").strip().lower()
    rationale = parsed.get("rationale")
    rationale = rationale.strip() if isinstance(rationale, str) and rationale.strip() else None
    if state == "question":
        question = parsed.get("question")
        if not isinstance(question, str) or not question.strip():
            return ClarifyOutcome(task_id, False, "error", "question response missing question")
        raw_choices = parsed.get("choices") or []
        choices = [str(c).strip() for c in raw_choices if str(c).strip()] if isinstance(raw_choices, list) else []
        return ClarifyOutcome(
            task_id,
            True,
            "question",
            question=question.strip(),
            choices=choices[:4],
            rationale=rationale,
        )
    if state == "ready":
        spec = parsed.get("spec_markdown")
        if not isinstance(spec, str) or not spec.strip():
            return ClarifyOutcome(task_id, False, "error", "ready response missing spec_markdown")
        return ClarifyOutcome(
            task_id,
            True,
            "ready",
            rationale=rationale,
            spec_markdown=spec.strip(),
        )
    return ClarifyOutcome(task_id, False, "error", f"unknown clarify state: {state!r}")


def _fallback_question(task_id: str, comments: list[kb.Comment]) -> ClarifyOutcome:
    answered = sum(1 for c in comments if (c.body or "").startswith(ANSWER_PREFIX))
    q = FALLBACK_QUESTIONS[min(answered, len(FALLBACK_QUESTIONS) - 1)]
    return ClarifyOutcome(
        task_id,
        True,
        "question",
        question=q["question"],
        choices=list(q["choices"]),
        rationale=q["rationale"],
        reason="fallback",
    )


def _call_aux(task: kb.Task, comments: list[kb.Comment], *, force_final: bool, timeout: Optional[int]) -> ClarifyOutcome:
    try:
        from agent.auxiliary_client import get_auxiliary_extra_body, get_text_auxiliary_client
    except Exception as exc:  # pragma: no cover
        logger.debug("clarify: auxiliary client import failed: %s", exc)
        return _fallback_question(task.id, comments)

    try:
        client, model = get_text_auxiliary_client("kanban_clarifier")
    except Exception as exc:
        logger.debug("clarify: get_text_auxiliary_client failed: %s", exc)
        return _fallback_question(task.id, comments)

    if client is None or not model:
        return _fallback_question(task.id, comments)

    user_msg = _USER_TEMPLATE.format(
        task_id=task.id,
        title=_truncate(task.title or "", 400),
        status=task.status,
        body=_truncate(task.body or "(no body)", 5000),
        transcript=_truncate(clarify_transcript(comments), 6000),
        instruction=(
            "Finalize a ready spec now. Do not ask another question unless the task would be unsafe without it."
            if force_final
            else "Ask the next question, or return ready if enough is known."
        ),
    )
    try:
        create_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=HERMES_KANBAN_CLARIFY_MAX_TOKENS,
            timeout=timeout or 120,
            extra_body=get_auxiliary_extra_body() or None,
        )
        if not is_truthy_value(os.getenv("HERMES_AUX_NO_TEMPERATURE", "")):
            create_kwargs["temperature"] = 0.2
        resp = client.chat.completions.create(**create_kwargs)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.info("clarify: API call failed for %s (%s) — using fallback", task.id, exc)
        return _fallback_question(task.id, comments)

    parsed = _extract_json_blob(raw)
    if parsed is None:
        return ClarifyOutcome(task.id, False, "error", "LLM returned malformed JSON")
    return _normalize(parsed, task.id)


def _question_comment(outcome: ClarifyOutcome) -> str:
    return QUESTION_PREFIX + ": " + json.dumps(
        {
            "state": "question",
            "question": outcome.question,
            "choices": outcome.choices,
            "rationale": outcome.rationale,
        },
        ensure_ascii=False,
    )


def _ready_comment(outcome: ClarifyOutcome) -> str:
    return READY_PREFIX + ": " + json.dumps(
        {
            "state": "ready",
            "spec_markdown": outcome.spec_markdown,
            "rationale": outcome.rationale,
        },
        ensure_ascii=False,
    )


def clarify_next(task_id: str, *, author: Optional[str] = None, timeout: Optional[int] = None) -> ClarifyOutcome:
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return ClarifyOutcome(task_id, False, "error", "unknown task id")
        if task.status != "triage":
            return ClarifyOutcome(task_id, False, "error", f"task is not in triage (status={task.status!r})")
        comments = kb.list_comments(conn, task_id)

    outcome = _call_aux(task, comments, force_final=False, timeout=timeout)
    if not outcome.ok:
        return outcome

    with kb.connect_closing() as conn:
        if outcome.state == "question":
            kb.add_comment(conn, task_id, author=author or "clarifier", body=_question_comment(outcome))
        elif outcome.state == "ready":
            kb.add_comment(conn, task_id, author=author or "clarifier", body=_ready_comment(outcome))
    return outcome


def record_answer(task_id: str, answer: str, *, author: Optional[str] = None) -> ClarifyOutcome:
    answer = (answer or "").strip()
    if not answer:
        return ClarifyOutcome(task_id, False, "error", "answer is required")
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return ClarifyOutcome(task_id, False, "error", "unknown task id")
        if task.status != "triage":
            return ClarifyOutcome(task_id, False, "error", f"task is not in triage (status={task.status!r})")
        kb.add_comment(
            conn,
            task_id,
            author=author or _profile_author(),
            body=ANSWER_PREFIX + ": " + answer,
        )
    return ClarifyOutcome(task_id, True, "answer", reason="recorded")


def finalize_spec(task_id: str, *, author: Optional[str] = None, timeout: Optional[int] = None) -> ClarifyOutcome:
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return ClarifyOutcome(task_id, False, "error", "unknown task id")
        if task.status != "triage":
            return ClarifyOutcome(task_id, False, "error", f"task is not in triage (status={task.status!r})")
        comments = kb.list_comments(conn, task_id)

    outcome = _call_aux(task, comments, force_final=True, timeout=timeout)
    if not outcome.ok:
        return outcome
    if outcome.state == "question":
        # The model refused to finalize because one more decision matters.
        with kb.connect_closing() as conn:
            kb.add_comment(conn, task_id, author=author or "clarifier", body=_question_comment(outcome))
        return outcome

    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET body = ? WHERE id = ? AND status = 'triage'",
                (outcome.spec_markdown, task_id),
            )
            if cur.rowcount != 1:
                return ClarifyOutcome(task_id, False, "error", "task moved out of triage before finalize")
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, strftime('%s','now'))",
                (task_id, (author or "clarifier"), _ready_comment(outcome)),
            )
    return outcome
