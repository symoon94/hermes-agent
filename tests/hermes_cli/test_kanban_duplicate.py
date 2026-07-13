from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_duplicate as kd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _response(payload: dict):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
    )


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        assert "EXISTING TASKS" in kwargs["messages"][0]["content"]
        return _response(self.payload)


def test_semantic_duplicate_returns_existing_ticket(monkeypatch, kanban_home):
    with kb.connect() as conn:
        existing_id = kb.create_task(conn, title="고객사 설치 오류 원인 분석", body="SolarBox 설치 실패 로그를 조사한다", priority=1)
        client = _Client({
            "duplicate": True,
            "reason": "동일한 설치 실패 원인 분석 작업입니다.",
            "matches": [{"id": existing_id, "confidence": 0.94, "reason": "대상과 산출물이 같습니다."}],
        })
        monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (client, "test-model"))
        decision = kd.check_duplicate(conn, title="SolarBox 고객 설치 실패 로그 조사")

    assert decision.available is True
    assert decision.duplicate is True
    assert decision.model == "test-model"
    assert decision.matches[0].id == existing_id
    assert decision.matches[0].confidence == 0.94


def test_duplicate_check_rejects_unknown_ids(monkeypatch, kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="실제 태스크", priority=1)
        client = _Client({
            "duplicate": True,
            "reason": "중복이라고 주장",
            "matches": [{"id": "t_invented", "confidence": 1.0, "reason": "환각"}],
        })
        monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (client, "test-model"))
        decision = kd.check_duplicate(conn, title="새 태스크")

    assert decision.available is True
    assert decision.duplicate is False
    assert decision.matches == []


def test_done_ticket_remains_a_duplicate_candidate(monkeypatch, kanban_home):
    with kb.connect() as conn:
        existing_id = kb.create_task(conn, title="completed outcome", priority=1)
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (existing_id,))
        client = _Client({
            "duplicate": True,
            "reason": "완료된 티켓과 같은 결과물입니다.",
            "matches": [{"id": existing_id, "confidence": 0.9, "reason": "same outcome"}],
        })
        monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (client, "test-model"))
        decision = kd.check_duplicate(conn, title="repeat completed outcome")

    assert decision.duplicate is True
    assert decision.matches[0].status == "done"


def test_malformed_llm_json_is_reported_unavailable(monkeypatch, kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="existing", priority=1)
        client = _Client({"duplicate": False, "matches": []})
        client.create = lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
        )
        monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (client, "test-model"))
        decision = kd.check_duplicate(conn, title="new")

    assert decision.available is False
    assert decision.source == "invalid-response"


def test_duplicate_check_surfaces_unavailable_llm(monkeypatch, kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="기존 태스크", priority=1)
        monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda _task: (None, None))
        decision = kd.check_duplicate(conn, title="새 태스크")

    assert decision.available is False
    assert decision.duplicate is False
    assert decision.source == "unavailable"


def test_empty_board_needs_no_llm(kanban_home):
    with kb.connect() as conn:
        decision = kd.check_duplicate(conn, title="첫 태스크")

    assert decision.available is True
    assert decision.duplicate is False
    assert decision.source == "empty-board"
