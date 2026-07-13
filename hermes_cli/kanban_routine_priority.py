"""Values-aware ordering for active Kanban routine checklist items."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoutineRankDecision:
    ordered_ids: list[str]
    reason: str
    source: str
    model: Optional[str] = None


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


def _fallback_order(items: list[dict[str, Any]]) -> list[str]:
    """Transparent practical fallback aligned with the user's stated values."""
    def score(item: dict[str, Any]) -> tuple[int, int, str]:
        text = f"{item.get('title') or ''} {item.get('body') or ''}".lower()
        if any(k in text for k in ("exercise", "운동", "health", "건강", "sleep", "수면")):
            tier = 1
        elif any(k in text for k in ("speaking", "writing", "english", "영어", "interview", "면접")):
            tier = 2
        elif any(k in text for k in ("reading", "learn", "study", "익히", "학습", "기술")):
            tier = 3
        elif any(k in text for k in ("weekly", "준비", "meeting", "회의")):
            tier = 4
        else:
            tier = 3
        return tier, int(item.get("sort_order") or 0), str(item.get("id") or "")

    return [item["id"] for item in sorted(items, key=score)]


def decide_routine_order(items: list[dict[str, Any]]) -> RoutineRankDecision:
    if not items:
        return RoutineRankDecision([], "활성 루틴이 없습니다.", "empty")
    if len(items) == 1:
        return RoutineRankDecision([items[0]["id"]], "활성 루틴이 하나입니다.", "single")

    fallback = _fallback_order(items)
    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, model = get_text_auxiliary_client("kanban_priority")
        if client is None or not model:
            return RoutineRankDecision(fallback, "우선순위 LLM unavailable; practical fallback", "heuristic")
        compact = [
            {
                "id": item["id"],
                "title": item.get("title") or "",
                "body": (item.get("body") or "")[:500],
                "frequency": item.get("frequency") or "daily",
                "checked_today": bool(item.get("checked_today")),
                "current_sort_order": int(item.get("sort_order") or 0),
            }
            for item in items
        ]
        prompt = (
            "Order Sooyoung Moon's recurring Kanban routines by the order they should be shown and attempted.\n"
            "Return every supplied id exactly once. This is ordering WITHIN each UI frequency group; daily and weekly remain visually separate.\n\n"
            "Evidence hierarchy:\n"
            "1. Health, sleep, family/relationship stability, safety, hard external commitments.\n"
            "2. Explicit career goal: overseas big-tech readiness, English technical communication, portfolio/interview assets.\n"
            "3. Automation, reusable systems, and learning that compounds or reduces repeated toil.\n"
            "4. Administrative meeting preparation and routine chores, unless deadline/impact makes them urgent.\n"
            "Prefer a sustainable routine over optimizing everything into obligation. Do not rank only by frequency or creation time.\n"
            "Use Jyotish only as a secondary planning signal, never fate: Sagittarius Lagna favors overseas/learning; Capricorn stellium favors durable systems; Cancer Moon/Ashlesha favors health, relationships, boundaries and hidden-complexity resolution; Saturn Aquarius favors technology/platform leverage; Venus/Jupiter period moderately supports language, partnership, career assets and quality of life.\n"
            "Return ONLY JSON: {\"ordered_ids\":[...],\"reason\":\"짧은 한국어 설명\"}.\n\n"
            "ROUTINES:\n" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content if resp and resp.choices else ""
        obj = _extract_json_object(raw)
        supplied = [item["id"] for item in items]
        ordered = obj.get("ordered_ids") if obj else None
        if not isinstance(ordered, list):
            return RoutineRankDecision(fallback, "LLM 응답 파싱 실패; practical fallback", "heuristic-fallback", model)
        ordered = [str(value) for value in ordered]
        if len(ordered) != len(supplied) or set(ordered) != set(supplied):
            return RoutineRankDecision(fallback, "LLM이 루틴 ID를 누락/추가함; practical fallback", "heuristic-fallback", model)
        return RoutineRankDecision(
            ordered,
            str(obj.get("reason") or "사용자 가치 기준으로 루틴을 정렬했습니다.")[:500],
            "auxiliary",
            model,
        )
    except Exception as exc:
        logger.info("routine priority LLM unavailable: %s", exc)
        return RoutineRankDecision(
            fallback,
            f"LLM unavailable ({type(exc).__name__}); practical fallback",
            "heuristic-fallback",
        )


def rerank_routines(conn: Any) -> RoutineRankDecision:
    from hermes_cli import kanban_db as kb

    routines = kb.list_routines(conn)
    items = [
        {
            "id": routine.id,
            "title": routine.title,
            "body": routine.body,
            "frequency": routine.frequency,
            "checked_today": routine.checked_today,
            "sort_order": routine.sort_order,
        }
        for routine in routines
    ]
    decision = decide_routine_order(items)
    now = __import__("time").time()
    with kb.write_txn(conn):
        for rank, routine_id in enumerate(decision.ordered_ids, start=1):
            conn.execute(
                "UPDATE routine_items SET sort_order=?, updated_at=? WHERE id=? AND active=1",
                (rank, int(now), routine_id),
            )
    return decision
