from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_duplicate as kd
from hermes_cli import kanban_intake as ki


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="new triage task", priority=0):
    return kb.create_task(conn, title=title, triage=True, priority=priority)


def test_unique_triage_task_is_ranked_and_marked_checked(monkeypatch, kanban_home):
    with kb.connect_closing() as conn:
        task_id = _create_triage(conn)
    monkeypatch.setattr(
        kd,
        "check_duplicate",
        lambda *a, **kw: kd.DuplicateDecision(True, False, [], "unique", "test"),
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_priority.decide_priority",
        lambda *a, **kw: __import__("hermes_cli.kanban_priority", fromlist=["PriorityDecision"]).PriorityDecision(
            2, "values aligned", tier=2, realm="Private", source="test"
        ),
    )

    result = ki.process_triage_task(task_id)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        kinds = [event.kind for event in kb.list_events(conn, task_id)]
        checked = ki.is_triage_duplicate_checked(conn, task_id)
    assert result["state"] == "unique"
    assert task is not None and task.status == "triage" and task.priority == 2
    assert "triage_duplicate_checked" in kinds
    assert checked is True


def test_checked_triage_task_does_not_call_llm_twice(monkeypatch, kanban_home):
    with kb.connect_closing() as conn:
        task_id = _create_triage(conn)
    calls = 0

    def unique(*args, **kwargs):
        nonlocal calls
        calls += 1
        return kd.DuplicateDecision(True, False, [], "unique", "test")

    monkeypatch.setattr(kd, "check_duplicate", unique)
    assert ki.process_triage_task(task_id)["state"] == "unique"
    assert ki.process_triage_task(task_id)["state"] == "already_checked"
    assert calls == 1


def test_duplicate_triage_task_is_archived_with_match(monkeypatch, kanban_home):
    with kb.connect_closing() as conn:
        existing = kb.create_task(conn, title="existing", priority=1)
        task_id = _create_triage(conn, title="same outcome")
    decision = kd.DuplicateDecision(
        True,
        True,
        [kd.DuplicateMatch(existing, "existing", "ready", 0.93, "same deliverable")],
        "duplicate",
        "test",
    )
    monkeypatch.setattr(kd, "check_duplicate", lambda *a, **kw: decision)

    result = ki.process_triage_task(task_id)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
    assert result["state"] == "duplicate_archived"
    assert task is not None and task.status == "archived" and task.priority == 0
    assert existing in comments[-1].body


def test_llm_failure_keeps_unmarked_task_in_triage_for_retry(monkeypatch, kanban_home):
    with kb.connect_closing() as conn:
        task_id = _create_triage(conn)
    monkeypatch.setattr(
        kd,
        "check_duplicate",
        lambda *a, **kw: kd.DuplicateDecision(False, False, [], "provider down", "test"),
    )

    result = ki.process_triage_task(task_id)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        checked = ki.is_triage_duplicate_checked(conn, task_id)
    assert result["state"] == "retry"
    assert task is not None and task.status == "triage"
    assert checked is False


def test_explicit_priority_survives_unique_gate(monkeypatch, kanban_home):
    with kb.connect_closing() as conn:
        task_id = _create_triage(conn, priority=7)
    monkeypatch.setattr(
        kd,
        "check_duplicate",
        lambda *a, **kw: kd.DuplicateDecision(True, False, [], "unique", "test"),
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_priority.decide_priority",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("explicit priority must not be reranked")),
    )

    result = ki.process_triage_task(task_id)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert result["state"] == "unique"
    assert task is not None and task.priority == 7


def test_duplicate_candidate_query_excludes_current_triage_task(kanban_home):
    with kb.connect_closing() as conn:
        existing = kb.create_task(conn, title="existing", priority=1)
        current = _create_triage(conn)
        candidates = kd._candidate_tasks(conn, exclude_task_id=current)
    ids = {item["id"] for item in candidates}
    assert existing in ids
    assert current not in ids
