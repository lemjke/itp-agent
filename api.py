"""REST API ИТП-агента: тот же агент и та же история диалогов, что в веб-интерфейсе.

    uvicorn api:app --port 8000          # документация: http://localhost:8000/docs
"""
from typing import Literal

import ollama
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from itp import history
from itp.agent import CONTEXT_MODES, ask
from itp.config import DB_DSN, EMBED_MODEL, LLM_HOST, LLM_MODEL

app = FastAPI(
    title="ИТП-агент API",
    description="Диагностика индивидуальных тепловых пунктов на локальной LLM: вопросы агенту, история "
                "диалогов, оценки ответов.",
    version="1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["Что не так с ИТП-3 за последнюю неделю?"])
    chat_id: str | None = Field(None, description="Продолжить диалог (уточнения вроде «а за январь?»); "
                                                  "без него начинается новый")
    context: Literal["rewrite", "messages", "none"] = Field(
        "rewrite", description="Как учитывать историю: " + "; ".join(f"{k} — {v}" for k, v in CONTEXT_MODES.items()))


class ToolCall(BaseModel):
    tool: str
    args: dict


class AskResponse(BaseModel):
    chat_id: str
    message_id: int
    asked: str
    understood_as: str = Field(description="Вопрос, на который агент отвечал (после учёта диалога)")
    answer: str
    findings: list[dict] = Field(description="Находки, подтверждённые детекторами")
    recommendations: list[str]
    sources: list[str] = Field(description="Пункты нормативных документов, найденные поиском")
    dropped_findings: int = Field(description="Сколько находок модели отброшено проверкой")
    unverified_numbers: list[str] = Field(description="Числа ответа, которых нет в данных инструментов")
    tool_calls: list[ToolCall]
    seconds: float


class Feedback(BaseModel):
    value: Literal[0, 1] | None = Field(description="1 — 👍, 0 — 👎, null — снять оценку")


def _response(chat_id, message_id, asked, result):
    report = result["report"]
    return AskResponse(
        chat_id=chat_id, message_id=message_id, asked=asked, understood_as=result.get("question", asked),
        answer=report["answer"], findings=report.get("findings", []),
        recommendations=report.get("recommendations", []), sources=report.get("sources", []),
        dropped_findings=result.get("dropped_findings", 0), unverified_numbers=result.get("unverified_numbers", []),
        tool_calls=[ToolCall(tool=s["tool"], args=s["args"]) for s in result["trace"]],
        seconds=result["seconds"])


@app.post("/ask", response_model=AskResponse, summary="Задать вопрос агенту")
def ask_agent(req: AskRequest):
    """Синхронно: ответ занимает 15–60 с (локальная LLM). Вопрос и ответ сохраняются в историю диалога."""
    if req.chat_id:
        past = history.load(req.chat_id)
        if not past:
            raise HTTPException(404, f"Диалог {req.chat_id} не найден")
        chat_id = req.chat_id
    else:
        past, chat_id = [], history.new_chat_id()
    turns = history.as_turns((m["question"], m["result"]) for m in past)
    try:
        result = ask(req.question, history=turns, context=req.context)
    except Exception as exc:
        raise HTTPException(503, f"Агент не смог ответить: {exc}. Запущен ли Ollama?") from exc
    message_id = history.save(chat_id, req.question, result, LLM_MODEL)
    return _response(chat_id, message_id, req.question, result)


@app.get("/chats", summary="Последние диалоги")
def list_chats(limit: int = 15):
    return history.chats(limit=max(1, min(limit, 100)))


@app.get("/chats/{chat_id}", summary="Сообщения диалога")
def get_chat(chat_id: str):
    rows = history.load(chat_id)
    if not rows:
        raise HTTPException(404, f"Диалог {chat_id} не найден")
    return {"chat_id": chat_id, "messages": [
        {**_response(chat_id, r["id"], r["question"], r["result"]).model_dump(), "feedback": r["feedback"]}
        for r in rows]}


@app.post("/messages/{message_id}/feedback", summary="Оценить ответ")
def feedback(message_id: int, body: Feedback):
    if not history.set_feedback(message_id, body.value):
        raise HTTPException(404, f"Сообщение {message_id} не найдено")
    return {"message_id": message_id, "feedback": body.value}


@app.get("/health", summary="Доступны ли БД и модели")
def health():
    status = {"db": False, "ollama": False, "models": {LLM_MODEL: False, EMBED_MODEL: False}}
    try:
        with psycopg.connect(DB_DSN, connect_timeout=3) as con:
            status["db"] = con.execute("SELECT count(*) FROM norm_chunks").fetchone()[0] > 0
    except psycopg.Error:
        pass
    try:
        names = {m.model for m in ollama.Client(host=LLM_HOST, timeout=3).list().models}
        status["ollama"] = True
        status["models"] = {m: m in names or f"{m}:latest" in names for m in status["models"]}
    except Exception:          # Ollama не запущен или недоступен по сети
        pass
    status["ok"] = status["db"] and status["ollama"] and all(status["models"].values())
    return status
