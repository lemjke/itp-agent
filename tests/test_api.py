"""REST API с подменёнными агентом и историей: без LLM и без БД."""
import pytest
from fastapi.testclient import TestClient

import api

REPORT = {"answer": "На ИТП-3 обратка выше графика.", "findings": [], "recommendations": [], "sources": []}


class FakeHistory:
    def __init__(self):
        self.rows, self.next_id = {}, 1

    def new_chat_id(self):
        return "chat1"

    def load(self, chat_id):
        return self.rows.get(chat_id, [])

    def save(self, chat_id, question, result, model, arch="react"):
        row = {"id": self.next_id, "question": question, "result": result, "feedback": None}
        self.rows.setdefault(chat_id, []).append(row)
        self.next_id += 1
        return row["id"]

    def chats(self, limit=15):
        return [{"chat_id": c, "title": r[0]["question"], "count": len(r)} for c, r in self.rows.items()]

    def set_feedback(self, message_id, value):
        for rows in self.rows.values():
            for r in rows:
                if r["id"] == message_id:
                    r["feedback"] = value
                    return True
        return False

    as_turns = staticmethod(api.history.as_turns)


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_ask(question, history=None, context="rewrite", **_):
        calls.append({"question": question, "history": history, "context": context})
        understood = "Что не так с ИТП-3 в январе?" if history else question
        return {"report": REPORT, "trace": [{"tool": "detect_anomalies", "args": {"substation": "ИТП-3"}}],
                "question": understood, "dropped_findings": 0, "unverified_numbers": [], "seconds": 1.0}

    monkeypatch.setattr(api, "history", FakeHistory())
    monkeypatch.setattr(api, "ask", fake_ask)
    c = TestClient(api.app)
    c.calls = calls
    return c


def test_new_chat_then_follow_up_passes_history(client):
    r1 = client.post("/ask", json={"question": "Что не так с ИТП-3 за последнюю неделю?"}).json()
    assert (r1["chat_id"], r1["message_id"], r1["tool_calls"][0]["tool"]) == ("chat1", 1, "detect_anomalies")
    assert client.calls[0]["history"] == []

    r2 = client.post("/ask", json={"question": "А за январь?", "chat_id": "chat1"}).json()
    assert r2["understood_as"] == "Что не так с ИТП-3 в январе?" and r2["asked"] == "А за январь?"
    assert client.calls[1]["history"] == [{"question": "Что не так с ИТП-3 за последнюю неделю?",
                                           "answer": REPORT["answer"]}]


def test_chat_listing_and_feedback(client):
    client.post("/ask", json={"question": "Проверь ИТП-1"})
    assert client.get("/chats").json()[0]["count"] == 1
    assert client.post("/messages/1/feedback", json={"value": 1}).json() == {"message_id": 1, "feedback": 1}
    assert client.get("/chats/chat1").json()["messages"][0]["feedback"] == 1


@pytest.mark.parametrize("method, url, body, code", [
    ("post", "/ask", {"question": ""}, 422),
    ("post", "/ask", {"question": "А за январь?", "chat_id": "нет такого"}, 404),
    ("post", "/ask", {"question": "Проверь ИТП-1", "context": "telepathy"}, 422),
    ("get", "/chats/нет-такого", None, 404),
    ("post", "/messages/999/feedback", {"value": 1}, 404),
    ("post", "/messages/1/feedback", {"value": 5}, 422),
])
def test_errors(client, method, url, body, code):
    resp = client.post(url, json=body) if method == "post" else client.get(url)
    assert resp.status_code == code


def test_agent_failure_is_503(client, monkeypatch):
    def broken(*_, **__):
        raise ConnectionError("Ollama недоступен")
    monkeypatch.setattr(api, "ask", broken)
    assert client.post("/ask", json={"question": "Проверь ИТП-1"}).status_code == 503
