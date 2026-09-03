"""
LLM-powered version of the Triage Agent using Groq (OpenAI-compatible) function calling.
"""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from tools import pattern_match, check_upi_id, check_phone_number, request_more_info

load_dotenv()
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_INSTRUCTION = """
You are a Scam/Fraud Triage Agent. A user will paste a suspicious SMS, call
transcript, or UPI request. Your job:

1. Investigate using the tools available to you (pattern_match, check_upi_id,
   check_phone_number). Call whichever tools are relevant.
2. If the evidence is inconclusive, call request_more_info with a specific
   clarifying question instead of guessing.
3. Once confident, call the submit_verdict tool with your final verdict,
   reason, and confidence. Do NOT write your final answer as plain text -
   always use the submit_verdict tool to conclude.

Never guess when evidence is weak - ask for more information instead.
"""

TOOL_FUNCTIONS = {
    "pattern_match": pattern_match,
    "check_upi_id": check_upi_id,
    "check_phone_number": check_phone_number,
    "request_more_info": request_more_info,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "pattern_match",
            "description": "Check message text against known scam pattern templates.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The message text to check"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_upi_id",
            "description": "Check a UPI ID against the reported-scam-IDs database.",
            "parameters": {
                "type": "object",
                "properties": {"upi_id": {"type": "string", "description": "The UPI ID to check, e.g. name@bank"}},
                "required": ["upi_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_phone_number",
            "description": "Check a phone number against the reported-scam-numbers database.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "string", "description": "The phone number to check"}},
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_more_info",
            "description": "Ask the user a clarifying question when evidence is inconclusive.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The clarifying question to ask"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": "Submit your final triage verdict once you are confident. Always use this to conclude - never answer in plain text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["BLOCK & REPORT", "CAUTION", "LIKELY SAFE"],
                        "description": "The final verdict",
                    },
                    "reason": {"type": "string", "description": "Explanation for the verdict"},
                    "confidence": {"type": "number", "description": "Confidence from 0.0 to 1.0"},
                },
                "required": ["verdict", "reason", "confidence"],
            },
        },
    },
]


class LLMTriageAgent:
    def __init__(self):
        self.trace = []
        self.messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    def run(self, user_input: str) -> dict:
        self.trace = []
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(5):  # cap iterations to avoid infinite loops
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                raw_text = (msg.content or "").strip()
                self.messages.append({"role": "assistant", "content": raw_text})
                return self._parse_final(raw_text)

            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self.trace.append({"step": f"act:{fn_name}", "detail": args})

                if fn_name == "submit_verdict":
                    return {
                        "action": "verdict",
                        "verdict": args.get("verdict", "CAUTION"),
                        "reason": args.get("reason", "No reason provided."),
                        "confidence": float(args.get("confidence", 0.5)),
                        "trace": self.trace,
                    }

                if fn_name == "request_more_info":
                    self.trace.append({"step": "result:request_more_info", "detail": {"action": "ask_user"}})
                    return {
                        "action": "ask_user",
                        "question": args.get("question", "Can you provide more detail?"),
                        "trace": self.trace,
                    }

                fn = TOOL_FUNCTIONS.get(fn_name)
                result = fn(**args) if fn else {"error": f"Unknown tool {fn_name}"}

                self.trace.append({"step": f"result:{fn_name}", "detail": result})

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        return {
            "action": "verdict",
            "verdict": "CAUTION",
            "reason": "Agent could not reach a confident conclusion within its step limit.",
            "confidence": 0.3,
            "trace": self.trace,
        }

    def _parse_final(self, raw_text: str) -> dict:
        try:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            return {
                "action": "verdict",
                "verdict": parsed.get("verdict", "CAUTION"),
                "reason": parsed.get("reason", raw_text),
                "confidence": float(parsed.get("confidence", 0.5)),
                "trace": self.trace,
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "action": "ask_user",
                "question": raw_text,
                "trace": self.trace,
            }


if __name__ == "__main__":
    agent = LLMTriageAgent()
    result = agent.run("Dear customer your account will be blocked. Update your KYC now: http://bit.ly/xyz123")
    print(json.dumps(result, indent=2))