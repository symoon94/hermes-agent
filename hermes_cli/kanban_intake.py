"""Triage intake gate: semantic duplicate detection before decomposition."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_duplicate, kanban_priority

logger = logging.getLogger(__name__)


def _decision_payload(decision: kanban_duplicate.DuplicateDecision) -> dict[str, Any]:
    return decision.to_dict()


def _latest_intake_event(conn: Any, task_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        """
        SELECT kind, payload
          FROM task_events
         WHERE task_id = ?
           AND kind IN ('triage_duplicate_checked', 'triage_duplicate_archived')
         ORDER BY id DESC LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["payload"] or "{}")
    except Exception:
        payload = {}
    return {"kind": row["kind"], "payload": payload}


def process_triage_task(task_id: str, *, board: Optional[str] = None) -> dict[str, Any]:
    """Check a Triage card exactly once before its normal AI decomposition.

    Duplicate cards are archived with an audit event/comment pointing to the
    strongest existing match. Unique cards remain in Triage and receive a
    ``triage_duplicate_checked`` marker; the gateway may then pass them to the
    existing auto-decomposer, which preserves the user's normal workflow.
    LLM failures leave the task in Triage and unmarked so a later tick retries.
    """
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return {"ok": False, "state": "missing", "task_id": task_id}
        if task.status != "triage":
            return {"ok": False, "state": "not_triage", "task_id": task_id, "status": task.status}

        previous = _latest_intake_event(conn, task.id)
        if previous and previous["kind"] == "triage_duplicate_checked":
            return {"ok": True, "state": "already_checked", "task_id": task.id}

        decision = kanban_duplicate.check_duplicate(
            conn,
            title=task.title,
            body=task.body,
            tenant=task.tenant,
            exclude_task_id=task.id,
        )
        payload = _decision_payload(decision)
        if not decision.available:
            # Keep it unmarked in Triage so the next gateway tick retries. Avoid
            # comment spam; the event trail is enough for diagnostics.
            with kb.write_txn(conn):
                kb._append_event(conn, task.id, "triage_duplicate_check_failed", payload)
            return {"ok": False, "state": "retry", "task_id": task.id, "decision": payload}

        if decision.duplicate:
            strongest = decision.matches[0] if decision.matches else None
            match_text = (
                f"{strongest.id} [{strongest.status}] {strongest.title} "
                f"({round(strongest.confidence * 100)}%)"
                if strongest else "unknown"
            )
            with kb.write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET status='archived', priority=0, claim_lock=NULL, "
                    "claim_expires=NULL, worker_pid=NULL WHERE id=? AND status='triage'",
                    (task.id,),
                )
                if cur.rowcount != 1:
                    return {"ok": False, "state": "raced", "task_id": task.id}
                kb._append_event(conn, task.id, "triage_duplicate_archived", payload)
                conn.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, strftime('%s','now'))",
                    (
                        task.id,
                        "triage-intake",
                        "TRIAGE_DUPLICATE: " + match_text + "\n" + decision.reason,
                    ),
                )
            return {"ok": True, "state": "duplicate_archived", "task_id": task.id, "decision": payload}

        priority_decision = None
        if int(task.priority or 0) <= 0:
            priority_decision = kanban_priority.decide_priority(
                conn,
                title=task.title,
                body=task.body,
                tenant=task.tenant,
            )

        with kb.write_txn(conn):
            row = conn.execute("SELECT status, priority FROM tasks WHERE id=?", (task.id,)).fetchone()
            if row is None or row["status"] != "triage":
                return {"ok": False, "state": "raced", "task_id": task.id}
            if priority_decision is not None:
                rank = int(priority_decision.priority)
                if rank > 0:
                    kanban_priority.shift_for_insert(conn, rank)
                conn.execute("UPDATE tasks SET priority=? WHERE id=?", (rank, task.id))
                payload["priority"] = {
                    "rank": rank,
                    "insert_before_id": priority_decision.insert_before_id,
                    "tier": priority_decision.tier,
                    "realm": priority_decision.realm,
                    "source": priority_decision.source,
                    "reason": priority_decision.reason,
                }
            kb._append_event(conn, task.id, "triage_duplicate_checked", payload)
        return {
            "ok": True,
            "state": "unique",
            "task_id": task.id,
            "decision": payload,
            "priority": payload.get("priority", {}).get("rank", int(task.priority or 0)),
        }


def is_triage_duplicate_checked(conn: Any, task_id: str) -> bool:
    event = _latest_intake_event(conn, task_id)
    return bool(event and event["kind"] == "triage_duplicate_checked")
