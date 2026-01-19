import os
import requests
from fastapi import FastAPI, Request

from db_setup import build_db
from chatbot_core import set_db, generate_reply

app = FastAPI()

BREVO_API_KEY = os.environ["BREVO_API_KEY"]


@app.on_event("startup")
def startup():
    # Build your vector DB once when the API starts
    db = build_db()
    set_db(db)


@app.post("/brevo-webhook")
async def brevo_webhook(req: Request):
    payload = await req.json()

    if payload.get("eventName") != "conversationFragment":
        return {"ok": True}

    visitor_id = (payload.get("visitor") or {}).get("id")
    messages = payload.get("messages") or []
    if not visitor_id or not messages:
        return {"ok": True}

    last = messages[-1]
    if last.get("type") != "visitor":
        return {"ok": True}

    text = (last.get("text") or "").strip()
    if not text:
        return {"ok": True}

    # ✅ RAG answer
    answer = generate_reply(text)

    headers = {"api-key": BREVO_API_KEY, "content-type": "application/json"}
    body = {"visitorId": visitor_id, "text": answer, "receivedFrom": "MyBot"}

    r = requests.post(
        "https://api.brevo.com/v3/conversations/messages",
        headers=headers,
        json=body,
        timeout=15,
    )
    return {"ok": r.ok, "status": r.status_code}

