"""
Minimal Flask API wrapper around your existing LLMTriageAgent / TriageAgent.
Run this alongside bot.py: `python api_server.py` (separate terminal from app.py).
Does not touch app.py or your Streamlit UI — this is a new, additional entry point.
"""

from flask import Flask, request, jsonify
from llm_agent import LLMTriageAgent
from agent import TriageAgent

app = Flask(__name__)

# One agent instance per Telegram/Discord user, so follow-up questions keep
# context — mirrors how app.py keeps one agent per Streamlit session.
agents = {}


def get_agent(session_id):
    if session_id not in agents:
        agents[session_id] = LLMTriageAgent()
    return agents[session_id]


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    session_id = data.get("session_id", "default")

    agent = get_agent(session_id)
    try:
        result = agent.run(text)
    except Exception as e:
        print(f"[api_server] LLMTriageAgent failed ({e}), falling back to TriageAgent")
        fallback = TriageAgent()
        result = fallback.run(text)

    if result.get("action") == "ask_user":
        return jsonify({
            "category": "UNCLEAR",
            "verdict": "I need a bit more information to be sure.",
            "followup_question": result["question"],
            "evidence_hint": None,
        })

    verdict_label = result.get("verdict", "CAUTION")  # BLOCK & REPORT / CAUTION / LIKELY SAFE
    category = "SCAM" if verdict_label in ("BLOCK & REPORT", "CAUTION") else "SAFE"
    verdict_text = f"{verdict_label}: {result.get('reason', '')}"

    return jsonify({
        "category": category,
        "verdict": verdict_text,
        "followup_question": None,
        "evidence_hint": "screenshot, sender number, transaction ID" if category == "SCAM" else None,
    })


if __name__ == "__main__":
    app.run(port=5000)