"""Value-aware priority insertion for Hermes kanban tasks.

Hermes stores task priority as an integer rank where 1 is the top of the
master priority list.  When callers omit a priority, this module compares the
new card against the existing ranked backlog and returns an insertion rank.

The ranking policy intentionally mirrors the Luna Vibe prioritizer:
  tier 1 = revenue/IPO/customer/security/legal/blocker
  tier 2 = career/startup/product/process/design/release/migration leverage
  tier 3 = learning/documentation/knowledge-transfer/personal growth
  tier 4 = reject/defer candidates

The LLM path is the source of truth when available because it can compare the
actual ticket text against the current master list and Sooyoung's values.  The
heuristic fallback exists only so capture still works when the auxiliary model
is unavailable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


UPSTAGE_HINTS = (
    "upstage", "enterprise portal", "엔터프라이즈 포탈", "엔터포탈",
    "solarbox", "solar box", "솔라박스", "ai pack", "aipack",
    "catalog", "카탈로그", "operator", "installer", "파트너", "고객",
    "customer", "delivery", "납품", "product", "제품", "jira", "slack",
)
PRIVATE_HINTS = (
    "personal", "private", "개인", "영어", "english", "이력서", "resume",
    "interview", "면접", "창업", "startup", "사이드프로젝트", "학습", "가족",
)
TIER1_HINTS = (
    "매출", "상장", "revenue", "sales", "고객 장애", "prod outage",
    "production outage", "보안", "security", "legal", "compliance",
    "blocker", "blocking", "긴급", "urgent", "장애", "incident", "vip",
)
TIER2_HINTS = (
    "정책", "policy", "설계", "design", "proposal", "계획", "plan",
    "portal", "manual", "매뉴얼", "release", "릴리스", "workflow",
    "워크플로우", "raci", "migration", "마이그레이션", "자동화", "automation",
    "팀", "team", "제품", "product", "고객", "partner", "파트너",
)
TIER3_HINTS = (
    "study", "studying", "learn", "learning", "research", "til",
    "학습", "리서치", "정리", "가이드", "guide", "faq", "문서", "book", "책",
)
TIER4_HINTS = ("거절", "reject", "defer", "won't do", "wont do", "나중에", "someday")


@dataclass
class PriorityDecision:
    priority: int
    reason: str
    insert_before_id: Optional[str] = None
    tier: Optional[int] = None
    realm: Optional[str] = None
    source: str = "heuristic"


def _text(title: str, body: Optional[str], tenant: Optional[str] = None) -> str:
    return " ".join(str(x or "") for x in (title, body, tenant)).lower()


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _heuristic_tier_realm(title: str, body: Optional[str], tenant: Optional[str] = None) -> tuple[int, Optional[str]]:
    text = _text(title, body, tenant)
    upstageish = _has_any(text, UPSTAGE_HINTS)
    privateish = _has_any(text, PRIVATE_HINTS)
    realm = "Upstage" if upstageish else "Private" if privateish else None
    if _has_any(text, TIER4_HINTS):
        tier = 4
    elif _has_any(text, TIER1_HINTS):
        tier = 1
    elif upstageish and _has_any(text, TIER2_HINTS):
        tier = 2
    elif privateish or _has_any(text, TIER3_HINTS):
        tier = 3
    elif upstageish:
        tier = 2
    else:
        tier = 3
    return tier, realm


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


def _existing_ranked(conn: Any, *, limit: int = 120) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, title, body, status, assignee, priority, created_at
          FROM tasks
         WHERE status != 'archived'
           AND COALESCE(priority, 0) > 0
         ORDER BY priority ASC, created_at ASC
         LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        body = row["body"] or ""
        out.append({
            "id": row["id"],
            "priority": int(row["priority"] or 0),
            "status": row["status"],
            "assignee": row["assignee"],
            "title": row["title"],
            "body_snippet": re.sub(r"\s+", " ", body).strip()[:240],
        })
    return out


def _max_rank(conn: Any) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(priority), 0) AS p FROM tasks WHERE status != 'archived' AND COALESCE(priority, 0) > 0"
    ).fetchone()
    return int(row["p"] if row and row["p"] is not None else 0)


def _priority_before(conn: Any, task_id: Optional[str]) -> Optional[int]:
    if not task_id:
        return None
    row = conn.execute(
        "SELECT priority FROM tasks WHERE id = ? AND status != 'archived'",
        (task_id,),
    ).fetchone()
    if not row:
        return None
    p = int(row["priority"] or 0)
    return p if p > 0 else None


def _fallback_decision(conn: Any, *, title: str, body: Optional[str], tenant: Optional[str]) -> PriorityDecision:
    tier, realm = _heuristic_tier_realm(title, body, tenant)
    existing = _existing_ranked(conn, limit=500)
    # Approximate Luna's tier ordering. We infer tiers for existing items from
    # text and insert after the last item with a strictly better/equal tier.
    insert_after_priority = 0
    for item in existing:
        etier, _ = _heuristic_tier_realm(item.get("title") or "", item.get("body_snippet") or "", None)
        if etier <= tier:
            insert_after_priority = max(insert_after_priority, int(item.get("priority") or 0))
    priority = insert_after_priority + 1 if insert_after_priority else 1
    return PriorityDecision(
        priority=priority,
        reason=f"heuristic Luna tier T{tier}; inserted after existing tasks with tier <= T{tier}",
        tier=tier,
        realm=realm,
        source="heuristic",
    )


def decide_priority(conn: Any, *, title: str, body: Optional[str], tenant: Optional[str] = None) -> PriorityDecision:
    """Return the master-list insertion rank for a new task.

    The returned priority may collide with existing rows; caller should shift
    existing priorities >= this value before inserting the new card.
    """
    existing = _existing_ranked(conn)
    if not existing:
        tier, realm = _heuristic_tier_realm(title, body, tenant)
        return PriorityDecision(
            priority=1,
            reason="first ranked task on this board",
            tier=tier,
            realm=realm,
            source="empty-board",
        )

    fallback = _fallback_decision(conn, title=title, body=body, tenant=tenant)
    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, aux_model = get_text_auxiliary_client("kanban_priority")
        if client is None or not aux_model:
            return fallback
        prompt = (
            "You are inserting a new Hermes kanban card into Sooyoung Moon's master priority list.\n"
            "Use Sooyoung/Luna priority values, not FIFO and not arbitrary numeric score:\n"
            "- T1: revenue/IPO/customer-impact/security/legal/prod blocker/urgent external commitment.\n"
            "- T2: career/startup leverage, product lifecycle, team/process/system design, release/migration work.\n"
            "- T3: learning, documentation, knowledge transfer, personal growth/routines.\n"
            "- T4: reject/defer/someday candidates.\n"
            "Within a tier compare actual impact, deadlines, dependencies, reversibility, and whether the task unblocks other people.\n"
            "Return ONLY JSON with keys: insert_before_id (task id string or null), tier (1-4), realm ('Upstage'|'Private'|null), reason (short Korean).\n"
            "If the new card belongs at the end of the ranked list, use insert_before_id:null.\n\n"
            f"NEW CARD:\nTitle: {title}\nTenant: {tenant or ''}\nBody:\n{(body or '')[:2000]}\n\n"
            "CURRENT MASTER PRIORITY LIST (top first):\n"
            + json.dumps(existing, ensure_ascii=False, indent=2)
        )
        resp = client.chat.completions.create(
            model=aux_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        obj = _extract_json_object(text)
        if not obj:
            return fallback
        before = obj.get("insert_before_id")
        if before in ("", "null", "None"):
            before = None
        before = str(before) if before else None
        p = _priority_before(conn, before)
        if p is None:
            p = _max_rank(conn) + 1
            before = None
        tier = obj.get("tier")
        try:
            tier_int = int(tier) if tier is not None else fallback.tier
        except Exception:
            tier_int = fallback.tier
        realm = obj.get("realm") or fallback.realm
        return PriorityDecision(
            priority=int(p),
            insert_before_id=before,
            tier=tier_int,
            realm=str(realm) if realm else None,
            reason=str(obj.get("reason") or fallback.reason)[:500],
            source="auxiliary",
        )
    except Exception as exc:
        return PriorityDecision(
            priority=fallback.priority,
            reason=f"{fallback.reason}; auxiliary unavailable: {type(exc).__name__}",
            tier=fallback.tier,
            realm=fallback.realm,
            source="heuristic-fallback",
        )


def shift_for_insert(conn: Any, priority: int) -> None:
    """Make room for a new task at priority by shifting ranked rows down."""
    priority = max(1, int(priority))
    conn.execute(
        """
        UPDATE tasks
           SET priority = priority + 1
         WHERE status != 'archived'
           AND COALESCE(priority, 0) >= ?
        """,
        (priority,),
    )
