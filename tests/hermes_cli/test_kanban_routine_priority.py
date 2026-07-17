from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_routine_priority as krp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class _Client:
    def __init__(self, content):
        self.content = content
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        assert "family/relationship stability" in kwargs["messages"][0]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def _items():
    return [
        {"id": "r_read", "title": "Reading", "frequency": "daily", "sort_order": 1},
        {"id": "r_ex", "title": "Exercise", "frequency": "daily", "sort_order": 2},
        {"id": "r_speak", "title": "English speaking", "frequency": "daily", "sort_order": 3},
    ]


def test_gpt_routine_order_validates_and_returns_all_ids(monkeypatch):
    client = _Client('{"ordered_ids":["r_ex","r_speak","r_read"],"reason":"건강, 영어, 학습 순"}')
    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (client, "gpt-test"))
    decision = krp.decide_routine_order(_items())
    assert decision.ordered_ids == ["r_ex", "r_speak", "r_read"]
    assert decision.source == "auxiliary"
    assert decision.model == "gpt-test"


def test_hallucinated_routine_id_uses_practical_fallback(monkeypatch):
    client = _Client('{"ordered_ids":["r_fake"],"reason":"bad"}')
    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (client, "gpt-test"))
    decision = krp.decide_routine_order(_items())
    assert decision.ordered_ids == ["r_ex", "r_speak", "r_read"]
    assert decision.source == "heuristic-fallback"


def test_invalid_primary_json_retries_with_gpt_fallback(monkeypatch):
    solar = _Client("Exercise, English speaking, Reading")
    gpt = _Client('{"ordered_ids":["r_ex","r_speak","r_read"],"reason":"GPT fallback 정렬"}')
    requested_tasks = []

    def get_client(task):
        requested_tasks.append(task)
        if task == "kanban_priority":
            return solar, "solar-open2"
        if task == "kanban_priority_fallback":
            return gpt, "gpt-5.5"
        raise AssertionError(f"unexpected auxiliary task: {task}")

    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", get_client)

    decision = krp.decide_routine_order(_items())

    assert requested_tasks == ["kanban_priority", "kanban_priority_fallback"]
    assert decision.ordered_ids == ["r_ex", "r_speak", "r_read"]
    assert decision.reason == "GPT fallback 정렬"
    assert decision.source == "auxiliary-fallback"
    assert decision.model == "gpt-5.5"


def test_rerank_persists_sort_order(monkeypatch, kanban_home):
    with kb.connect_closing() as conn:
        read = kb.create_routine_item(conn, title="Reading", frequency="daily")
        exercise = kb.create_routine_item(conn, title="Exercise", frequency="daily")
        speaking = kb.create_routine_item(conn, title="English speaking", frequency="daily")
        monkeypatch.setattr(
            krp,
            "decide_routine_order",
            lambda _items: krp.RoutineRankDecision([exercise, speaking, read], "test", "test"),
        )
        krp.rerank_routines(conn)
        ordered = kb.list_routines(conn)
    assert [item.id for item in ordered] == [exercise, speaking, read]
    assert [item.sort_order for item in ordered] == [1, 2, 3]


def test_fallback_prioritizes_health_then_english_then_learning():
    assert krp._fallback_order(_items()) == ["r_ex", "r_speak", "r_read"]
