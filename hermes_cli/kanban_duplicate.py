"""LLM-based semantic duplicate detection for Hermes Kanban task creation."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DuplicateMatch:
    id: str
    title: str
    status: str
    confidence: float
    reason: str


@dataclass
class DuplicateDecision:
    available: bool
    duplicate: bool
    matches: list[DuplicateMatch]
    reason: str
    source: str
    model: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "duplicate": self.duplicate,
            "matches": [asdict(match) for match in self.matches],
            "reason": self.reason,
            "source": self.source,
            "model": self.model,
        }


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    text = text or ""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fenced or re.findall(r"\{.*\}", text, re.DOTALL)
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _candidate_tasks(
    conn: Any,
    *,
    limit: int = 240,
    exclude_task_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return compact active-task context, ranked work first then recent work."""
    rows = conn.execute(
        """
        SELECT id, title, body, status, priority, tenant, created_at
          FROM tasks
         WHERE status != 'archived'
           AND (? IS NULL OR id != ?)
         ORDER BY CASE WHEN COALESCE(priority, 0) > 0 THEN 0 ELSE 1 END,
                  CASE WHEN COALESCE(priority, 0) > 0 THEN priority END ASC,
                  created_at DESC
         LIMIT ?
        """,
        (exclude_task_id, exclude_task_id, max(1, int(limit))),
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        body = re.sub(r"\s+", " ", row["body"] or "").strip()
        candidates.append(
            {
                "id": row["id"],
                "title": row["title"],
                "status": row["status"],
                "priority": int(row["priority"] or 0),
                "tenant": row["tenant"],
                "body_snippet": body[:320],
            }
        )
    return candidates


def check_duplicate(
    conn: Any,
    *,
    title: str,
    body: Optional[str] = None,
    tenant: Optional[str] = None,
    exclude_task_id: Optional[str] = None,
) -> DuplicateDecision:
    """Ask the configured auxiliary LLM whether the proposed task is a duplicate.

    This intentionally has no lexical fallback: if the LLM is unavailable, the
    caller receives ``available=False`` and must surface that uncertainty to the
    user instead of silently pretending a semantic check occurred.
    """
    candidates = _candidate_tasks(conn, exclude_task_id=exclude_task_id)
    if not candidates:
        return DuplicateDecision(True, False, [], "보드에 비교할 기존 태스크가 없습니다.", "empty-board")

    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, model = get_text_auxiliary_client("kanban_duplicate")
        if client is None or not model:
            return DuplicateDecision(False, False, [], "중복 검사 LLM을 사용할 수 없습니다.", "unavailable")

        prompt = (
            "You are the semantic duplicate gate for Sooyoung Moon's Hermes Kanban board.\n"
            "Decide whether the proposed task represents substantially the SAME intended deliverable, "
            "problem, or action as an existing ticket. Compare meaning, not wording.\n\n"
            "Rules:\n"
            "- Similar topic alone is NOT a duplicate if scope, target, deliverable, incident, or time window differs.\n"
            "- Rephrased titles or detailed-vs-short descriptions ARE duplicates when they ask for the same outcome.\n"
            "- A done ticket may still be a duplicate; mention its status so the user can choose whether recurrence is intentional.\n"
            "- Return at most 3 strongest matches. Do not invent task IDs.\n"
            "- duplicate=true only when at least one match has confidence >= 0.78.\n"
            "- reason and per-match reasons must be concise Korean.\n"
            "Return ONLY JSON: {\"duplicate\":bool,\"reason\":str,\"matches\":["
            "{\"id\":str,\"confidence\":0..1,\"reason\":str}]}.\n\n"
            f"PROPOSED TASK:\nTitle/detail:\n{title[:6000]}\n"
            f"Additional body:\n{(body or '')[:3000]}\nTenant: {tenant or ''}\n\n"
            "EXISTING TASKS:\n"
            + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # Reasoning models may spend a substantial hidden-token budget
            # before emitting the compact JSON answer.
            max_tokens=8000,
        )
        raw = resp.choices[0].message.content if resp and resp.choices else ""
        obj = _extract_json_object(raw)
        if not obj:
            return DuplicateDecision(False, False, [], "중복 검사 LLM 응답을 해석하지 못했습니다.", "invalid-response", model)

        by_id = {item["id"]: item for item in candidates}
        matches: list[DuplicateMatch] = []
        for item in obj.get("matches") or []:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("id") or "")
            candidate = by_id.get(task_id)
            if not candidate:
                continue
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
            except Exception:
                confidence = 0.0
            matches.append(
                DuplicateMatch(
                    id=task_id,
                    title=str(candidate.get("title") or ""),
                    status=str(candidate.get("status") or ""),
                    confidence=confidence,
                    reason=str(item.get("reason") or "의미적으로 유사함")[:500],
                )
            )
        matches.sort(key=lambda match: match.confidence, reverse=True)
        matches = matches[:3]
        duplicate = bool(obj.get("duplicate")) and any(match.confidence >= 0.78 for match in matches)
        return DuplicateDecision(
            True,
            duplicate,
            matches,
            str(obj.get("reason") or ("중복 후보가 있습니다." if duplicate else "의미상 중복이 아닙니다."))[:500],
            "auxiliary",
            model,
        )
    except Exception as exc:
        logger.info("kanban duplicate check unavailable: %s", exc)
        return DuplicateDecision(
            False,
            False,
            [],
            f"중복 검사 LLM 호출 실패: {type(exc).__name__}: {str(exc)[:300]}",
            "error",
        )
