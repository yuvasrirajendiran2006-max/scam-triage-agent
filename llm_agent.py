"""
LLM-powered Scam/Fraud Triage Agent using Groq
(OpenAI-compatible function calling).
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from tools import (
    pattern_match,
    check_upi_id,
    check_phone_number,
    request_more_info,
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "openai/gpt-oss-120b"

MAX_ITERATIONS = 4


# ============================================================
# 2. SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are a Scam/Fraud Triage Agent.

Your job is to analyze a suspicious SMS, call transcript, payment request,
UPI request, phone number, or other fraud-related message.

You must make the best possible decision using the information already
provided by the user.

There are THREE final verdicts:

1. BLOCK & REPORT
2. CAUTION
3. LIKELY SAFE

There is also a REQUEST_MORE_INFO action, but it must be used ONLY when
the input is genuinely too vague to perform any meaningful assessment.


============================================================
DECISION RULE 1 — BLOCK & REPORT
============================================================

Use BLOCK & REPORT when there is strong evidence of fraud or a strong
known scam pattern.

Examples include:

- OTP requests
- PIN or password requests
- Requests for banking credentials
- Remote-access or screen-sharing requests
- Fake KYC requests
- Fake account-blocking messages
- Lottery or prize scams requiring payment
- Fake refund scams
- Suspicious or malicious links
- Known scam UPI IDs
- Known scam phone numbers
- Clear phishing
- Clear impersonation
- Requests to install suspicious applications
- Requests to pay a verification fee
- Requests to pay a refund-processing fee
- Strong threats or pressure to make a payment


============================================================
DECISION RULE 2 — CAUTION
============================================================

Use CAUTION when there are suspicious indicators but there is NOT enough
evidence to prove that the message is definitely a scam.

Examples:

- A friend contacts the user from a new number and asks for money.
- Someone urgently asks the user to send money.
- An unfamiliar person asks for a payment.
- A payment request looks unusual but there is no confirmed scam evidence.
- Someone asks the user to verify something but there is not enough
  evidence to classify it as a definite scam.


IMPORTANT:

Suspicious does NOT automatically mean BLOCK & REPORT.

If the situation is suspicious but uncertain, use CAUTION.


============================================================
DECISION RULE 3 — LIKELY SAFE
============================================================

Use LIKELY SAFE when the message appears normal and contains no meaningful
scam indicators.

Examples:

- Normal delivery notifications
- Normal order updates
- Normal payment confirmations
- Ordinary conversations
- Routine service notifications
- Normal reminders without suspicious payment or credential requests


============================================================
DECISION RULE 4 — REQUEST MORE INFO
============================================================

Use REQUEST_MORE_INFO ONLY when the input is genuinely too vague to make
ANY meaningful assessment.

Example:

"Someone contacted me."

This is too vague.

Therefore:

REQUEST_MORE_INFO


But:

"My friend contacted me from a new number and urgently asked me to send
₹3000."

This is NOT too vague.

The correct decision is:

CAUTION


IMPORTANT:

DO NOT request more information simply because:

- a phone number is missing
- a UPI ID is missing
- a sender name is missing
- a URL is missing
- some optional information is missing

If enough information exists to identify suspicious behavior, give a
CAUTION verdict instead of asking a question.


============================================================
TOOL USAGE
============================================================

Use the available tools when they are relevant.

1. pattern_match

Use this when the message contains suspicious wording such as:

- OTP
- KYC
- account blocked
- refund
- prize
- lottery
- urgent payment
- threats
- suspicious links
- verification fee
- payment request
- remote access


2. check_upi_id

Use this when a UPI ID is present in the user input.

Do NOT invent a UPI ID.


3. check_phone_number

Use this when a phone number is present in the user input.

Do NOT invent a phone number.


4. request_more_info

Use this ONLY when the input is genuinely too vague to perform
a useful assessment.


============================================================
DECISION PRIORITY
============================================================

Follow this priority:

STRONG SCAM EVIDENCE
        ↓
BLOCK & REPORT

SUSPICIOUS BUT UNCERTAIN
        ↓
CAUTION

CLEARLY NORMAL / LEGITIMATE
        ↓
LIKELY SAFE

TOO VAGUE TO ASSESS
        ↓
REQUEST_MORE_INFO


============================================================
IMPORTANT SAFETY RULES
============================================================

Never invent:

- phone numbers
- UPI IDs
- URLs
- report counts
- names
- organizations
- scam database results

Only use information actually provided by the user or returned by tools.

Do not assume that every unfamiliar phone number is a scam.

Do not assume that every unfamiliar UPI ID is a scam.

Use tools to verify information when possible.

If a tool confirms that a UPI ID or phone number is reported as a scam,
this is strong evidence supporting BLOCK & REPORT.

If tools find no known scam evidence but the message itself contains
suspicious behavior, use CAUTION rather than automatically asking for
more information.


============================================================
FINAL RESPONSE RULE
============================================================

When enough information exists to make a decision:

ALWAYS call submit_verdict.

Do NOT provide the final verdict as ordinary text.

The submit_verdict tool must contain:

- verdict
- reason
- confidence

Confidence must be between 0.0 and 1.0.


============================================================
IMPORTANT EXAMPLES
============================================================

Example 1:

User:
"Your bank account will be blocked today. Send your OTP to complete KYC."

Decision:
BLOCK & REPORT


Example 2:

User:
"My friend contacted me from a new number and urgently asked me to send
₹3000."

Decision:
CAUTION

Do NOT ask for the phone number first.


Example 3:

User:
"Someone contacted me."

Decision:
REQUEST_MORE_INFO


Example 4:

User:
"Your order has been delivered successfully."

Decision:
LIKELY SAFE


Example 5:

User:
"You won ₹10 lakh in a lottery. Pay ₹500 processing fee to claim your prize."

Decision:
BLOCK & REPORT


Example 6:

User:
"Send the refund to refund.help@oksbi to receive your money."

If the UPI ID is reported by the tool/database:

Decision:
BLOCK & REPORT


Remember:

Do not ask unnecessary questions.

Use CAUTION when something is suspicious but not confirmed.

Use REQUEST_MORE_INFO only when the input is genuinely too vague.
"""


# ============================================================
# 3. TOOL FUNCTIONS
# ============================================================

TOOL_FUNCTIONS = {
    "pattern_match": pattern_match,
    "check_upi_id": check_upi_id,
    "check_phone_number": check_phone_number,
    "request_more_info": request_more_info,
}


# ============================================================
# 4. TOOL SCHEMAS
# ============================================================

TOOLS_SCHEMA = [

    # --------------------------------------------------------
    # PATTERN MATCH
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "pattern_match",
            "description": """
Check the user's message against known scam pattern templates.

Use this when the message contains suspicious wording such as OTP,
KYC, account blocking, refund, lottery, prize, urgent payment,
threats, suspicious links, verification fees, or similar scam signals.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact message text to check.",
                    }
                },
                "required": ["text"],
            },
        },
    },

    # --------------------------------------------------------
    # UPI CHECK
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "check_upi_id",
            "description": """
Check a UPI ID against the reported-scam-IDs database.

Only use this when the user has actually provided a UPI ID.
Never invent a UPI ID.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "upi_id": {
                        "type": "string",
                        "description": "The UPI ID provided by the user, e.g. name@bank.",
                    }
                },
                "required": ["upi_id"],
            },
        },
    },

    # --------------------------------------------------------
    # PHONE CHECK
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "check_phone_number",
            "description": """
Check a phone number against the reported-scam-numbers database.

Only use this when the user has actually provided a phone number.
Never invent a phone number.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "The phone number provided by the user.",
                    }
                },
                "required": ["number"],
            },
        },
    },

    # --------------------------------------------------------
    # REQUEST MORE INFO
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "request_more_info",
            "description": """
Ask the user a clarifying question ONLY when the input is genuinely too
vague to perform any meaningful fraud assessment.

Do NOT use this tool simply because a phone number, UPI ID, sender name,
or other optional information is missing.

If the situation is suspicious but uncertain, submit a CAUTION verdict
instead.

For example:

"My friend contacted me from a new number and urgently asked me to send
₹3000."

This should result in CAUTION, not REQUEST_MORE_INFO.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A specific clarifying question.",
                    }
                },
                "required": ["question"],
            },
        },
    },

    # --------------------------------------------------------
    # SUBMIT VERDICT
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": """
Submit the final scam triage verdict.

Use:

BLOCK & REPORT
when there is strong evidence of fraud.

CAUTION
when there are suspicious indicators but fraud is not confirmed.

LIKELY SAFE
when the message appears normal and has no meaningful scam indicators.

Always use this tool to conclude the analysis when enough information
is available.
""",
            "parameters": {
                "type": "object",
                "properties": {

                    "verdict": {
                        "type": "string",
                        "enum": [
                            "BLOCK & REPORT",
                            "CAUTION",
                            "LIKELY SAFE",
                        ],
                        "description": "The final fraud triage verdict.",
                    },

                    "reason": {
                        "type": "string",
                        "description": "Short explanation supporting the verdict.",
                    },

                    "confidence": {
                        "type": "number",
                        "description": "Confidence between 0.0 and 1.0.",
                    },
                },

                "required": [
                    "verdict",
                    "reason",
                    "confidence",
                ],
            },
        },
    },
]


# ============================================================
# 5. LLM TRIAGE AGENT
# ============================================================

class LLMTriageAgent:

    def __init__(self):
        self.trace = []


    # ========================================================
    # RUN AGENT
    # ========================================================

    def run(self, user_input: str) -> dict:

        # Reset trace for every new investigation
        self.trace = []

        # IMPORTANT:
        # Every input gets a fresh conversation.
        # This prevents previous test cases from affecting new ones.
        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]


        # ----------------------------------------------------
        # AGENT LOOP
        # ----------------------------------------------------

        for iteration in range(MAX_ITERATIONS):

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )

            msg = response.choices[0].message


            # ------------------------------------------------
            # NO TOOL CALL
            # ------------------------------------------------

            if not msg.tool_calls:

                raw_text = (msg.content or "").strip()

                messages.append({
                    "role": "assistant",
                    "content": raw_text,
                })

                return self._parse_final(raw_text)


            # ------------------------------------------------
            # SAVE ASSISTANT TOOL CALL
            # ------------------------------------------------

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    tc.model_dump()
                    for tc in msg.tool_calls
                ],
            })


            # ------------------------------------------------
            # PROCESS EACH TOOL CALL
            # ------------------------------------------------

            for tc in msg.tool_calls:

                fn_name = tc.function.name


                # --------------------------------------------
                # PARSE ARGUMENTS
                # --------------------------------------------

                try:
                    args = json.loads(
                        tc.function.arguments
                    )

                except json.JSONDecodeError:

                    args = {}


                # --------------------------------------------
                # TRACE ACTION
                # --------------------------------------------

                self.trace.append({
                    "step": f"act:{fn_name}",
                    "detail": args,
                })


                # ==================================================
                # FINAL VERDICT
                # ==================================================

                if fn_name == "submit_verdict":

                    verdict = args.get(
                        "verdict",
                        "CAUTION",
                    )

                    reason = args.get(
                        "reason",
                        "No reason provided.",
                    )

                    try:
                        confidence = float(
                            args.get(
                                "confidence",
                                0.5,
                            )
                        )

                    except (TypeError, ValueError):

                        confidence = 0.5


                    # Keep confidence between 0 and 1
                    confidence = max(
                        0.0,
                        min(1.0, confidence),
                    )


                    return {
                        "action": "verdict",
                        "verdict": verdict,
                        "reason": reason,
                        "confidence": confidence,
                        "trace": self.trace,
                    }


                # ==================================================
                # REQUEST MORE INFO
                # ==================================================

                if fn_name == "request_more_info":

                    question = args.get(
                        "question",
                        "Can you provide more details about the message?",
                    )

                    self.trace.append({
                        "step": "result:request_more_info",
                        "detail": {
                            "action": "ask_user",
                        },
                    })


                    return {
                        "action": "ask_user",
                        "question": question,
                        "trace": self.trace,
                    }


                # ==================================================
                # NORMAL TOOL
                # ==================================================

                fn = TOOL_FUNCTIONS.get(fn_name)


                if fn:

                    try:

                        result = fn(**args)

                    except Exception as e:

                        result = {
                            "error": str(e),
                        }

                else:

                    result = {
                        "error": f"Unknown tool {fn_name}"
                    }


                # --------------------------------------------
                # SAVE TOOL RESULT TO TRACE
                # --------------------------------------------

                self.trace.append({
                    "step": f"result:{fn_name}",
                    "detail": result,
                })


                # --------------------------------------------
                # SEND TOOL RESULT BACK TO LLM
                # --------------------------------------------

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })


        # ====================================================
        # FALLBACK
        # ====================================================

        return {
            "action": "verdict",
            "verdict": "CAUTION",
            "reason": (
                "The available information suggests caution, "
                "but the agent could not complete the analysis "
                "within the allowed number of steps."
            ),
            "confidence": 0.3,
            "trace": self.trace,
        }


    # ========================================================
    # PARSE FINAL RESPONSE
    # ========================================================

    def _parse_final(self, raw_text: str) -> dict:

        try:

            cleaned = (
                raw_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            parsed = json.loads(cleaned)


            verdict = parsed.get(
                "verdict",
                "CAUTION",
            )

            reason = parsed.get(
                "reason",
                raw_text,
            )


            try:

                confidence = float(
                    parsed.get(
                        "confidence",
                        0.5,
                    )
                )

            except (TypeError, ValueError):

                confidence = 0.5


            confidence = max(
                0.0,
                min(1.0, confidence),
            )


            return {
                "action": "verdict",
                "verdict": verdict,
                "reason": reason,
                "confidence": confidence,
                "trace": self.trace,
            }


        except (json.JSONDecodeError, ValueError, TypeError):

            # Do NOT ask the user another question just because
            # the LLM returned malformed plain text.
            #
            # Instead, use a safe fallback verdict.

            return {
                "action": "verdict",
                "verdict": "CAUTION",
                "reason": (
                    "The message could not be confidently classified. "
                    "Please verify the sender before taking any action."
                ),
                "confidence": 0.3,
                "trace": self.trace,
            }


# ============================================================
# 6. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    agent = LLMTriageAgent()

    test_input = (
        "Dear customer your account will be blocked. "
        "Update your KYC now: http://bit.ly/xyz123"
    )

    result = agent.run(test_input)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )