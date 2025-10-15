import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

def push(text: str):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            },
            timeout=5
        )
    except Exception as e:
        print(f"Push failed: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Unknown question: {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Record user contact info when they provide an email.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "User email"},
            "name": {"type": "string", "description": "User name"},
            "notes": {"type": "string", "description": "Conversation context"},
        },
        "required": ["email"],
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record questions that the bot could not answer.",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]
