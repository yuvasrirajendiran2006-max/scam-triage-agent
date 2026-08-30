"""
Tool functions for the Scam/Fraud Triage Agent.
Each function is a standalone, testable "tool" the agent can decide to call.
"""
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(DATA_DIR, "scam_patterns.json")) as f:
    SCAM_PATTERNS = json.load(f)

with open(os.path.join(DATA_DIR, "reported_ids.json")) as f:
    REPORTED = json.load(f)


def pattern_match(text: str) -> dict:
    """
    Check the message text against known scam pattern templates.
    Returns the best match (if any) with a confidence score.
    """
    text_lower = text.lower()
    matches = []
    for pattern in SCAM_PATTERNS:
        hit_count = sum(1 for kw in pattern["keywords"] if kw in text_lower)
        if hit_count > 0:
            confidence = min(1.0, hit_count / max(2, len(pattern["keywords"]) // 2))
            matches.append({
                "category": pattern["category"],
                "risk": pattern["risk"],
                "explanation": pattern["explanation"],
                "confidence": round(confidence, 2),
                "keyword_hits": hit_count,
            })

    if not matches:
        return {"found": False, "confidence": 0.0, "matches": []}

    matches.sort(key=lambda m: m["confidence"], reverse=True)
    return {"found": True, "confidence": matches[0]["confidence"], "matches": matches}


def extract_upi_ids(text: str) -> list:
    """Pull anything that looks like a UPI ID (name@bank) out of free text."""
    return re.findall(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}", text)


def extract_phone_numbers(text: str) -> list:
    """Pull anything that looks like an Indian phone number out of free text."""
    return re.findall(r"(?:\+91[\-\s]?)?[6-9]\d{9}", text)


def check_upi_id(upi_id: str) -> dict:
    """Check a UPI ID against the reported-scam-IDs knowledge base."""
    for entry in REPORTED["upi_ids"]:
        if entry["id"].lower() == upi_id.lower():
            return {"found": True, "reports": entry["reports"], "note": entry["note"]}
    return {"found": False, "reports": 0, "note": "No reports found in local database."}


def check_phone_number(number: str) -> dict:
    """Check a phone number against the reported-scam-numbers knowledge base."""
    normalized = number.replace(" ", "").replace("-", "")
    for entry in REPORTED["phone_numbers"]:
        if entry["number"].replace(" ", "") in normalized or normalized in entry["number"]:
            return {"found": True, "reports": entry["reports"], "note": entry["note"]}
    return {"found": False, "reports": 0, "note": "No reports found in local database."}


def request_more_info(question: str) -> dict:
    """
    The agent calls this when it doesn't have enough signal to give a verdict.
    In the CLI/Streamlit demo this pauses and asks the user directly.
    """
    return {"action": "ask_user", "question": question}


TOOL_REGISTRY = {
    "pattern_match": pattern_match,
    "check_upi_id": check_upi_id,
    "check_phone_number": check_phone_number,
    "request_more_info": request_more_info,
}