"""Kanban team assembly — plan role-based multi-agent collaboration.

This is the Hermes-kanban equivalent of a team-builder/team-assemble pass.
It does not spawn workers directly. It writes a durable ``TEAM_ASSEMBLY``
comment that later decomposer/orchestrator passes can read as routing and
handoff guidance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod
from utils import env_int, is_truthy_value

logger = logging.getLogger(__name__)

TEAM_PREFIX = "TEAM_ASSEMBLY"
HERMES_KANBAN_TEAM_MAX_TOKENS = max(1500, env_int("HERMES_KANBAN_TEAM_MAX_TOKENS", 6000))

_SYSTEM_PROMPT = """You are the team assembler for a Hermes Kanban board.

Given a triage task and available Hermes profiles, decide whether the work should
be handled by one worker or by a role-based team. Your output is a plan that will
be stored as a TEAM_ASSEMBLY comment and used by the decomposer/orchestrator.

Return STRICT JSON only with this shape:
{
  "mode": "solo" | "small-team" | "fleet",
  "rationale": "why this team shape is appropriate",
  "roles": [
    {
      "role": "tech-lead | researcher | implementer | reviewer | docs | ops | ...",
      "profile": "existing profile name, or null if missing",
      "responsibility": "what this role owns",
      "deliverable": "what this role must hand off",
      "acceptance_criteria": ["verifiable condition", "..."]
    }
  ],
  "missing_roles": ["roles/profiles that would help but do not exist"],
  "handoff_protocol": ["how agents communicate through comments/results"],
  "review_strategy": "how output should be challenged/reviewed",
  "decomposition_hints": ["child task suggestions / dependency notes"],
  "risks": ["coordination or scope risks"]
}

Rules:
- Prefer solo for small, clear tasks. Do not create a team just because you can.
- Use small-team when different skills should work independently or in sequence.
- Use fleet only for many similar targets that can be processed in parallel.
- Pick profiles ONLY from the roster by matching descriptions. Use null when no
  available profile fits; list that gap in missing_roles.
- This is an asynchronous kanban team, not a live chat room. Handoffs happen via
  task body, comments, result summaries, blocked/review states, and parent-child
  links.
- Include a reviewer/challenge role when the work is risky, broad, or code-changing.
- Treat clarify transcript and human answers as authoritative.
- Do not invent repo paths, test results, credentials, PR numbers, or external facts.
- No markdown fences. Output only JSON.
"""

_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Clarify transcript:
{clarify_transcript}

Available profiles:
{roster}
"""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass
class TeamOutcome:
    task_id: str
    ok: bool
    reason: str = ""
    mode: str = "solo"
    roles: list[dict[str, Any]] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_json_blob(raw: str) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last <= first:
        return None
    try:
        val = json.loads(stripped[first:last + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return val if isinstance(val, dict) else None


def _profile_author() -> str:
    return os.environ.get("HERMES_PROFILE") or os.environ.get("USER") or "team-assembler"


def build_roster() -> tuple[list[dict[str, Any]], set[str]]:
    roster: list[dict[str, Any]] = []
    valid: set[str] = set()
    try:
        profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("team: failed to list profiles: %s", exc)
        return roster, valid
    for p in profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
        valid.add(p.name)
    return roster, valid


def format_roster(roster: list[dict[str, Any]]) -> str:
    if not roster:
        return "  (no profiles installed)"
    return "\n".join(
        f"  - {r['name']}{'' if r.get('has_description') else ' ⚠ undescribed'}: {r['description']}"
        for r in roster
    )


def latest_team_plan(comments: list[kb.Comment]) -> Optional[dict[str, Any]]:
    for c in reversed(comments):
        body = c.body or ""
        if not body.startswith(TEAM_PREFIX):
            continue
        payload = body[len(TEAM_PREFIX):].strip()
        if payload.startswith(":"):
            payload = payload[1:].strip()
        parsed = _extract_json_blob(payload)
        if parsed:
            return parsed
    return None


def team_transcript(comments: list[kb.Comment]) -> str:
    plan = latest_team_plan(comments)
    if not plan:
        return "(none yet)"
    return json.dumps(plan, ensure_ascii=False, indent=2)


def _clarify_transcript(comments: list[kb.Comment]) -> str:
    try:
        from hermes_cli import kanban_clarify
        return kanban_clarify.clarify_transcript(comments)
    except Exception:
        return "(none yet)"


def _normalise_plan(task_id: str, parsed: dict[str, Any], valid_profiles: set[str]) -> TeamOutcome:
    mode = str(parsed.get("mode") or "solo").strip().lower()
    if mode not in {"solo", "small-team", "fleet"}:
        mode = "solo"
    roles = parsed.get("roles") if isinstance(parsed.get("roles"), list) else []
    clean_roles: list[dict[str, Any]] = []
    missing: list[str] = []
    for r in roles:
        if not isinstance(r, dict):
            continue
        profile = r.get("profile")
        if isinstance(profile, str) and profile.strip() in valid_profiles:
            profile = profile.strip()
        else:
            if profile:
                missing.append(str(profile))
            profile = None
        role = str(r.get("role") or "worker").strip() or "worker"
        clean_roles.append({
            "role": role,
            "profile": profile,
            "responsibility": str(r.get("responsibility") or "").strip(),
            "deliverable": str(r.get("deliverable") or "").strip(),
            "acceptance_criteria": [str(x).strip() for x in (r.get("acceptance_criteria") or []) if str(x).strip()]
            if isinstance(r.get("acceptance_criteria") or [], list) else [],
        })
    for m in parsed.get("missing_roles") or []:
        if str(m).strip():
            missing.append(str(m).strip())
    plan = {
        "mode": mode,
        "rationale": str(parsed.get("rationale") or "").strip(),
        "roles": clean_roles,
        "missing_roles": sorted(set(missing)),
        "handoff_protocol": [str(x).strip() for x in (parsed.get("handoff_protocol") or []) if str(x).strip()]
        if isinstance(parsed.get("handoff_protocol") or [], list) else [],
        "review_strategy": str(parsed.get("review_strategy") or "").strip(),
        "decomposition_hints": [str(x).strip() for x in (parsed.get("decomposition_hints") or []) if str(x).strip()]
        if isinstance(parsed.get("decomposition_hints") or [], list) else [],
        "risks": [str(x).strip() for x in (parsed.get("risks") or []) if str(x).strip()]
        if isinstance(parsed.get("risks") or [], list) else [],
    }
    return TeamOutcome(task_id, True, "assembled", mode=mode, roles=clean_roles, missing_roles=plan["missing_roles"], plan=plan)


def _fallback_plan(task: kb.Task, roster: list[dict[str, Any]], valid_profiles: set[str]) -> TeamOutcome:
    active = None
    try:
        active = profiles_mod.get_active_profile_name() or "default"
    except Exception:
        active = "default"
    if active not in valid_profiles:
        active = next(iter(valid_profiles), None)
    parsed = {
        "mode": "solo",
        "rationale": "No team assembler model is configured; defaulting to one worker plus optional human review.",
        "roles": [{
            "role": "implementer",
            "profile": active,
            "responsibility": "Complete the clarified task end-to-end.",
            "deliverable": "Working result with verification summary.",
            "acceptance_criteria": ["Satisfy the task acceptance criteria", "Report verification evidence"],
        }],
        "missing_roles": [],
        "handoff_protocol": ["Use task comments/results for handoff; block if human input is required."],
        "review_strategy": "Use human review or a reviewer profile if the change is risky.",
        "decomposition_hints": ["Keep as a single task unless the spec spans independent workstreams."],
        "risks": ["Fallback plan may miss specialized profiles because no team model ran."],
    }
    return _normalise_plan(task.id, parsed, valid_profiles)


def _comment_body(plan: dict[str, Any]) -> str:
    return TEAM_PREFIX + ": " + json.dumps(plan, ensure_ascii=False)


def assemble_team(task_id: str, *, author: Optional[str] = None, timeout: Optional[int] = None) -> TeamOutcome:
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id) if task is not None else []
    if task is None:
        return TeamOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return TeamOutcome(task_id, False, f"task is not in triage (status={task.status!r})")

    roster, valid = build_roster()
    try:
        from agent.auxiliary_client import get_auxiliary_extra_body, get_text_auxiliary_client
        client, model = get_text_auxiliary_client("kanban_team_assembler")
    except Exception as exc:
        logger.debug("team: auxiliary unavailable: %s", exc)
        client = None
        model = None
    if client is None or not model:
        outcome = _fallback_plan(task, roster, valid)
    else:
        user_msg = _USER_TEMPLATE.format(
            task_id=task.id,
            title=_truncate(task.title or "", 400),
            body=_truncate(task.body or "(no body)", 5000),
            clarify_transcript=_truncate(_clarify_transcript(comments), 5000),
            roster=format_roster(roster),
        )
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                max_tokens=HERMES_KANBAN_TEAM_MAX_TOKENS,
                timeout=timeout or 120,
                extra_body=get_auxiliary_extra_body() or None,
            )
            if not is_truthy_value(os.getenv("HERMES_AUX_NO_TEMPERATURE", "")):
                kwargs["temperature"] = 0.2
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _extract_json_blob(raw)
            if parsed is None:
                return TeamOutcome(task_id, False, "LLM returned malformed JSON")
            outcome = _normalise_plan(task_id, parsed, valid)
        except Exception as exc:
            logger.info("team: API call failed for %s (%s)", task_id, exc)
            return TeamOutcome(task_id, False, f"LLM error: {type(exc).__name__}")

    with kb.connect_closing() as conn:
        kb.add_comment(conn, task_id, author=author or _profile_author(), body=_comment_body(outcome.plan))
    return outcome
