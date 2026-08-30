# Scam / Fraud Triage Agent

An agentic system that investigates a suspicious SMS, call transcript, or UPI
request and gives a clear act-now recommendation: **Block & Report**,
**Caution**, or **Likely Safe**.

Built for the Agentic AI Hackathon (Tech Zephyr 4.0, IIT Bhubaneswar).

## Why this needs to be agentic

A single LLM call or a fixed rule-list can't handle this well because:
- Scam wording constantly evolves — a static rule list goes stale
- The right investigation path differs per message (sometimes the text alone
  is enough, sometimes you need to check a UPI ID or number against a
  reputation database)
- When the first check is inconclusive, the system needs to **decide what to
  check next** or **ask the user a clarifying question** — not just guess

## Agent loop

```
Observe (user input)
   -> Decide + Act: pattern_match(text)
   -> Evaluate: confidence high? -> Verdict
   -> Adapt: confidence low? -> check_upi_id / check_phone_number
   -> Evaluate: still ambiguous? -> request_more_info(question)
   -> Re-run with follow-up answer -> Verdict
```

## Architecture

```
User Input (SMS / call transcript / UPI ID)
        |
        v
  +----------------+
  | Agent/Controller|  <-- agent.py: TriageAgent
  +----------------+
        |
        |--> Tool: pattern_match()        --> data/scam_patterns.json
        |--> Tool: check_upi_id()         --> data/reported_ids.json
        |--> Tool: check_phone_number()   --> data/reported_ids.json
        |--> Tool: request_more_info()    --> asks user, loops back in
        |
        v
   Verdict (Block & Report / Caution / Likely Safe) + reasoning trace
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or test the agent loop directly without any UI:
```bash
python3 agent.py
```

## Known limitations

- `data/reported_ids.json` is a seed dataset for demo purposes, not a live
  crowdsourced database.
- Current pattern matching uses keywords — misses paraphrased scams. Upgrading
  to an LLM-based planner (Gemini function calling) is the next step.

## Repo structure

```
scam_triage_agent/
|-- agent.py
|-- tools.py
|-- app.py
|-- requirements.txt
|-- .env.example
|-- README.md
|-- data/
|   |-- scam_patterns.json
|   |-- reported_ids.json
```