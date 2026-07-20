from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_priority as kp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_critical_due_date_boosts_within_importance_guardrail(kanban_home):
    now = 1_700_000_000
    with kb.connect() as conn:
        kb.create_task(conn, title="security incident", priority=1)
        kb.create_task(conn, title="read architecture book", priority=2)

        decision = kp._fallback_decision(
            conn,
            title="study evaluation methods",
            body=None,
            tenant=None,
            due_at=now + 12 * 60 * 60,
            now=now,
        )

    assert decision.priority == 2
    assert decision.tier == 2
    assert "critical" in decision.reason


def test_urgency_position_counts_better_tasks_even_if_existing_order_is_mixed(kanban_home):
    now = 1_700_000_000
    with kb.connect() as conn:
        kb.create_task(conn, title="security incident", priority=1)
        kb.create_task(conn, title="read architecture book", priority=2)
        kb.create_task(conn, title="Upstage release migration", priority=3)

        decision = kp._fallback_decision(
            conn,
            title="study evaluation methods",
            body=None,
            tenant=None,
            due_at=now + 12 * 60 * 60,
            now=now,
        )

    assert decision.priority == 3


class _Response:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class _Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        return _Response('{"insert_before_id":null,"tier":2,"realm":"Private","reason":"due-aware"}')


class _Client:
    def __init__(self):
        self.calls = []
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(self)


def test_llm_prompt_contains_structured_due_urgency(monkeypatch, kanban_home):
    now = 1_700_000_000
    due_at = now + 2 * 24 * 60 * 60
    client = _Client()
    monkeypatch.setattr(kp.time, "time", lambda: now)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda task: (client, "solar-open2"),
    )

    with kb.connect() as conn:
        kb.create_task(conn, title="existing product task", priority=1, due_at=now + 6 * 24 * 60 * 60)
        kp.decide_priority(
            conn,
            title="submit application",
            body="career opportunity",
            tenant="Private",
            due_at=due_at,
        )

    prompt = client.calls[0]["messages"][0]["content"]
    assert f"Due at unix: {due_at}" in prompt
    assert f"Current unix time: {now}" in prompt
    assert "promote by at most one tier" in prompt
    assert '"due_at": 1700518400' in prompt
    assert "AI-first, evaluation-driven, security-conscious, human-centered" in prompt
    assert "token economics, model and agent architecture, harness and evaluation, memory" in prompt
    assert "beneficial and easy to apply" in prompt


def test_future_aligned_ai_system_work_is_career_leverage_tier():
    tier, realm = kp._heuristic_tier_realm(
        "Build token-cost evaluation harness and agent memory guardrails",
        "Make AI safer and easier for people to adopt responsibly",
        None,
    )

    assert tier == 2
    assert realm is None


def test_refresh_due_urgency_reorders_once_when_time_bucket_changes(kanban_home):
    now = 1_700_000_000
    with kb.connect() as conn:
        security = kb.create_task(conn, title="security incident", priority=1)
        reading = kb.create_task(conn, title="read architecture book", priority=2)
        studying = kb.create_task(
            conn,
            title="study evaluation methods",
            priority=3,
            due_at=now + 8 * 24 * 60 * 60,
        )

        assert kp.refresh_due_urgency(conn, now=now) == 0
        assert kp.refresh_due_urgency(conn, now=now + 6 * 24 * 60 * 60) == 1
        ids = [task.id for task in kb.list_tasks(conn)]
        assert ids.index(security) < ids.index(studying) < ids.index(reading)
        assert kp.refresh_due_urgency(conn, now=now + 6 * 24 * 60 * 60) == 0
