import os
import requests
from fastapi import FastAPI, Request
from openai import OpenAI

app = FastAPI()
client = OpenAI()  # uses OPENAI_API_KEY from environment :contentReference[oaicite:2]{index=2}

BREVO_API_KEY = os.environ["BREVO_API_KEY"]

def bot_answer(user_text: str) -> str:
    resp = client.responses.create(
        model="gpt-4.1-mini",  # you can change to a model you have access to :contentReference[oaicite:3]{index=3}
        input=user_text,
        instructions="You are a helpful support assistant. Reply briefly and clearly.",
    )
    return resp.output_text

@app.post("/brevo-webhook")
async def brevo_webhook(req: Request):
    payload = await req.json()

    # Brevo Conversations message event :contentReference[oaicite:4]{index=4}
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

    answer = bot_answer(text)

    headers = {"api-key": BREVO_API_KEY, "content-type": "application/json"}  # :contentReference[oaicite:5]{index=5}
    body = {
        "visitorId": visitor_id,
        "text": answer,
        "receivedFrom": "MyBot",
    }

    r = requests.post("https://api.brevo.com/v3/conversations/messages", headers=headers, json=body, timeout=15)
    return {"ok": r.ok, "status": r.status_code}  # endpoint docs :contentReference[oaicite:6]{index=6}
