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
import time
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
    "여행", "쉼", "휴식", "건강", "운동", "즐기", "big tech", "빅테크",
    "overseas", "해외", "global", "글로벌",
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
    "big tech", "빅테크", "overseas", "해외", "global", "글로벌",
    "resume", "이력서", "interview", "면접", "portfolio", "포트폴리오",
    # Practical expressions of the user's secondary Jyotish alignment:
    # durable systems/technology (Capricorn + Saturn Aquarius), precise
    # communication/partnership (Mercury/Venus/Jupiter + Libra D9), and work
    # that protects the explicitly stated real-life values.
    "platform", "플랫폼", "infrastructure", "인프라", "system", "시스템",
    "reusable", "재사용", "communication", "커뮤니케이션", "language", "언어",
    "negotiation", "협상", "partnership", "협업", "role", "역할",
    "family", "가족", "health", "건강", "relationship", "관계",
    "energy", "에너지", "stability", "안정",
)
TIER3_HINTS = (
    "study", "studying", "learn", "learning", "research", "til",
    "학습", "리서치", "정리", "가이드", "guide", "faq", "문서", "book", "책",
)
TIER4_HINTS = ("거절", "reject", "defer", "won't do", "wont do", "나중에", "someday")
FUTURE_ALIGNMENT_HINTS = (
    "token economics", "token cost", "token-cost", "토큰 비용", "토큰 가성비",
    "model routing", "모델 라우팅", "agent system", "에이전트 시스템",
    "evaluation harness", "평가 하네스", "agent harness", "에이전트 하네스",
    "agent memory", "에이전트 메모리", "guardrail", "가드레일",
    "ai adoption", "ai 도입", "ai enablement", "production ai", "프로덕션 ai",
    "human-in-the-loop", "critical thinking", "비판적 사고",
    "easy-to-apply ai", "beneficial ai",
)


def due_urgency(due_at: Optional[int], *, now: Optional[int] = None) -> tuple[str, int]:
    """Return a stable urgency label and bucket (lower means more urgent)."""
    if due_at is None or int(due_at) <= 0:
        return "none", 3
    remaining = int(due_at) - int(time.time() if now is None else now)
    if remaining <= 24 * 60 * 60:
        return "critical", 0
    if remaining <= 3 * 24 * 60 * 60:
        return "near", 1
    if remaining <= 7 * 24 * 60 * 60:
        return "soon", 2
    return "later", 3


def _urgency_priority_key(tier: int, due_at: Optional[int], *, now: int) -> tuple[int, int, int]:
    label, bucket = due_urgency(due_at, now=now)
    effective_tier = tier
    if tier < 4 and label in {"critical", "near"}:
        effective_tier = max(1, tier - 1)
    # Original importance breaks ties before urgency, so a rushed low-value
    # task cannot leapfrog an intrinsically higher-tier task.
    return effective_tier, tier, bucket


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
    elif _has_any(text, FUTURE_ALIGNMENT_HINTS):
        tier = 2
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
        SELECT id, title, body, status, assignee, priority, created_at, due_at
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
            "due_at": int(row["due_at"]) if row["due_at"] is not None else None,
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


def _fallback_decision(
    conn: Any,
    *,
    title: str,
    body: Optional[str],
    tenant: Optional[str],
    due_at: Optional[int] = None,
    now: Optional[int] = None,
    exclude_task_id: Optional[str] = None,
) -> PriorityDecision:
    tier, realm = _heuristic_tier_realm(title, body, tenant)
    now_ts = int(time.time() if now is None else now)
    urgency_label, _ = due_urgency(due_at, now=now_ts)
    priority_key = _urgency_priority_key(tier, due_at, now=now_ts)
    effective_tier = priority_key[0]
    existing = _existing_ranked(conn, limit=500)
    if exclude_task_id:
        existing = [item for item in existing if item.get("id") != exclude_task_id]
    # Approximate Luna's tier ordering. Count every existing card with a
    # strictly better/equal guarded key so mixed legacy ordering cannot let a
    # lower-value urgent card jump an intrinsically higher-tier card.
    better_or_equal_count = 0
    for item in existing:
        etier, _ = _heuristic_tier_realm(item.get("title") or "", item.get("body_snippet") or "", None)
        existing_key = _urgency_priority_key(etier, item.get("due_at"), now=now_ts)
        if existing_key <= priority_key:
            better_or_equal_count += 1
    priority = better_or_equal_count + 1
    return PriorityDecision(
        priority=priority,
        reason=(
            f"heuristic Luna tier T{tier}, due urgency {urgency_label}; "
            f"effective tier T{effective_tier} with intrinsic-tier guardrail"
        ),
        tier=effective_tier,
        realm=realm,
        source="heuristic",
    )


def reprioritize_due_task(
    conn: Any,
    task_id: str,
    *,
    now: Optional[int] = None,
) -> Optional[PriorityDecision]:
    """Move an existing card using deterministic, importance-guarded urgency."""
    row = conn.execute(
        "SELECT id, title, body, tenant, due_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    due_at = int(row["due_at"]) if row["due_at"] is not None else None
    decision = _fallback_decision(
        conn,
        title=row["title"],
        body=row["body"],
        tenant=row["tenant"],
        due_at=due_at,
        now=now,
        exclude_task_id=task_id,
    )
    ranked_ids = [
        item["id"]
        for item in _existing_ranked(conn, limit=5000)
        if item["id"] != task_id
    ]
    target_index = max(0, min(len(ranked_ids), int(decision.priority) - 1))
    ranked_ids.insert(target_index, task_id)

    from hermes_cli import kanban_db

    event_time = int(time.time() if now is None else now)
    urgency_label, _ = due_urgency(due_at, now=event_time)
    with kanban_db.write_txn(conn):
        for priority, ranked_id in enumerate(ranked_ids, start=1):
            conn.execute(
                "UPDATE tasks SET priority = ? WHERE id = ?",
                (priority, ranked_id),
            )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'due_urgency_applied', ?, ?)",
            (
                task_id,
                json.dumps(
                    {
                        "priority": target_index + 1,
                        "tier": decision.tier,
                        "reason": decision.reason,
                        "urgency": urgency_label,
                    },
                    ensure_ascii=False,
                ),
                event_time,
            ),
        )
    return PriorityDecision(
        priority=target_index + 1,
        reason=decision.reason,
        tier=decision.tier,
        realm=decision.realm,
        source="due-urgency",
    )


def refresh_due_urgency(conn: Any, *, now: Optional[int] = None) -> int:
    """Re-rank cards once when their due date enters a more urgent bucket."""
    now_ts = int(time.time() if now is None else now)
    rows = conn.execute(
        """
        SELECT id, due_at
          FROM tasks
         WHERE due_at IS NOT NULL
           AND COALESCE(priority, 0) > 0
           AND status NOT IN ('done', 'archived')
         ORDER BY priority ASC, created_at ASC
        """
    ).fetchall()
    changed = 0
    for row in rows:
        urgency_label, _ = due_urgency(row["due_at"], now=now_ts)
        if urgency_label in {"none", "later"}:
            continue
        latest = conn.execute(
            """
            SELECT payload
              FROM task_events
             WHERE task_id = ? AND kind = 'due_urgency_applied'
             ORDER BY id DESC
             LIMIT 1
            """,
            (row["id"],),
        ).fetchone()
        previous_label = None
        if latest is not None and latest["payload"]:
            try:
                previous_label = json.loads(latest["payload"]).get("urgency")
            except (TypeError, ValueError, AttributeError):
                previous_label = None
        if previous_label == urgency_label:
            continue
        if reprioritize_due_task(conn, row["id"], now=now_ts) is not None:
            changed += 1
    return changed


def decide_priority(
    conn: Any,
    *,
    title: str,
    body: Optional[str],
    tenant: Optional[str] = None,
    due_at: Optional[int] = None,
) -> PriorityDecision:
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

    fallback = _fallback_decision(
        conn, title=title, body=body, tenant=tenant, due_at=due_at,
    )
    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, aux_model = get_text_auxiliary_client("kanban_priority")
        if client is None or not aux_model:
            return fallback
        now_ts = int(time.time())
        prompt_existing = []
        for item in existing:
            enriched = dict(item)
            label, _ = due_urgency(item.get("due_at"), now=now_ts)
            enriched["due_urgency"] = label
            enriched["due_in_hours"] = (
                round((int(item["due_at"]) - now_ts) / 3600, 1)
                if item.get("due_at") is not None else None
            )
            prompt_existing.append(enriched)
        prompt = (
            "You are inserting a new Hermes kanban card into Sooyoung Moon's master priority list.\n"
            "Use Sooyoung/Luna priority values, not FIFO and not arbitrary numeric score:\n"
            "- T1: revenue/IPO/customer-impact/security/legal/prod blocker/urgent external commitment.\n"
            "- T2: career/startup leverage, product lifecycle, team/process/system design, release/migration work.\n"
            "- T3: learning, documentation, knowledge transfer, personal growth/routines.\n"
            "- T4: reject/defer/someday candidates.\n"
            "Structured due-date urgency rules:\n"
            "- overdue or due within 24 hours = critical; within 3 days = near; within 7 days = soon.\n"
            "- Urgency may promote by at most one tier, and T4 defer/reject work is never promoted.\n"
            "- Preserve the intrinsic-importance guardrail: a rushed lower-value task must not leapfrog an intrinsically higher-tier task.\n"
            "- Within the same intrinsic importance, earlier deadlines, blocked people, and hard external commitments rank first.\n"
            "North-star future alignment — AI-first, evaluation-driven, security-conscious, human-centered:\n"
            "- The expected future is that every domain adopts AI and many jobs become meta-level orchestration: people specify, route, evaluate, govern, and take responsibility for AI work.\n"
            "- Strongly value compounding expertise in token economics, model and agent architecture, harness and evaluation, memory, guardrails, security, observability, and production quality.\n"
            "- Strongly value work that makes AI beneficial and easy to apply where it is absent or difficult to adopt, especially reusable enablement rather than one-off demos.\n"
            "- Strongly value rapid comprehension plus critical judgment of AI output, and human strengths such as communication, trust, empathy, facilitation, and accountability.\n"
            "- Health, family, stability, relationships, and recovery are enabling foundations for this future, not misaligned distractions.\n"
            "- Do not force a fake AI connection onto every chore: rank work by its real direct contribution, enabling contribution, hard obligation, or opportunity cost.\n"
            "Current personal value updates to weigh explicitly:\n"
            "- Strong desire to enjoy life, not optimize everything into obligation. Tasks that preserve energy, relationships, health, or daily enjoyment can outrank pure productivity chores.\n"
            "- Strong desire to work at an overseas big-tech company at least once. Tasks that build credible global-big-tech/FDE/product-engineering readiness, English communication, resume/portfolio, interview loops, or high-signal career options should rank higher.\n"
            "- Automation is intrinsically fun/energizing. Automation tasks get a positive boost when they also reduce repeated toil, improve team leverage, or create reusable agent/system capability.\n"
            "Vedic/Jyotish self-analysis to use as a SECONDARY planning signal (never as fate or as a reason to override real deadlines, customer impact, family/health, finances, or safety):\n"
            "- Sidereal D1 Sagittarius Lagna: expansion, overseas exposure, learning, teaching, and globally portable work are especially aligned.\n"
            "- Strong Capricorn stellium (Sun/Mercury/Venus/Mars): values durable systems, disciplined execution, measurable long-term assets, responsibility, and practical leverage; avoid busywork with no compounding result.\n"
            "- Cancer Moon in Ashlesha: strong strategic/relational sensitivity and pattern-reading; protect emotional energy and prefer work that resolves hidden complexity without creating unhealthy entanglement.\n"
            "- Jupiter in Libra and D9 Libra Ascendant: partnership, balance, negotiation, design judgment, and fair role structures matter; tasks improving collaboration or explicit agreements get a positive signal.\n"
            "- Saturn in Aquarius: long-horizon technology, systems, communities, and infrastructure are aligned, especially reusable platforms over one-off toil.\n"
            "- D9 strengths (Venus Taurus, Mercury Virgo, Jupiter Pisces): favor quality, communication precision, service, craft, and work that combines practicality with humane impact.\n"
            "- Current Venus Mahadasha (2015-2035): give a moderate positive signal to career/wealth-building, partnership, language/communication, quality of life, and family-supporting work—but timing is only advisory.\n"
            "Translate these chart signals into practical value alignment, not supernatural certainty.\n"
            "Within a tier compare actual impact, deadlines, dependencies, reversibility, and whether the task unblocks other people.\n"
            "Return ONLY JSON with keys: insert_before_id (task id string or null), tier (1-4), realm ('Upstage'|'Private'|null), reason (short Korean).\n"
            "If the new card belongs at the end of the ranked list, use insert_before_id:null.\n\n"
            f"Current unix time: {now_ts}\n"
            f"NEW CARD:\nTitle: {title}\nTenant: {tenant or ''}\n"
            f"Due at unix: {int(due_at) if due_at is not None else 'null'}\n"
            f"Due urgency: {due_urgency(due_at, now=now_ts)[0]}\n"
            f"Body:\n{(body or '')[:2000]}\n\n"
            "CURRENT MASTER PRIORITY LIST (top first):\n"
            + json.dumps(prompt_existing, ensure_ascii=False, indent=2)
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
