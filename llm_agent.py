"""
On-device Scam/Fraud Triage Agent (fast version).

- Offline database checks run first (instant).
- Plain confirmations (no link, no request) are marked LIKELY SAFE instantly.
- Clear scams get their verdict straight from the local database.
- Messages with no scam signals at all are marked LIKELY SAFE instantly.
- Only unclear messages go to the small local LLM (one short call).
- Advice text is pre-translated into Indian languages (no slow translation).
- Uses a local OpenAI-compatible server (Ollama / Qualcomm AI Hub).
  No cloud, no API key, works offline.
"""

import os
import re
import json
from openai import OpenAI

from tools import (
    pattern_match,
    check_upi_id,
    check_phone_number,
    extract_upi_ids,
    extract_phone_numbers,
)

# ------------------------------------------------------------
# 1. LOCAL LLM CLIENT
# ------------------------------------------------------------

client = OpenAI(
    api_key="local",
    base_url=os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1"),
    timeout=30,
    max_retries=0,
)

MODEL_NAME = os.environ.get("LOCAL_LLM_MODEL", "llama3.2:1b")

MAX_TOKENS = 70

VALID_VERDICTS = ("BLOCK & REPORT", "CAUTION", "LIKELY SAFE")


# ------------------------------------------------------------
# 2. PRE-TRANSLATED TEXT (instant, no model needed)
# ------------------------------------------------------------

VERDICT_LABELS = {
    "Tamil": {
        "BLOCK & REPORT": "தடுத்து புகாரளிக்கவும்",
        "CAUTION": "எச்சரிக்கை",
        "LIKELY SAFE": "பாதுகாப்பானது போல் தெரிகிறது",
    },
    "Hindi": {
        "BLOCK & REPORT": "ब्लॉक करें और रिपोर्ट करें",
        "CAUTION": "सावधान रहें",
        "LIKELY SAFE": "संभवतः सुरक्षित",
    },
    "Telugu": {
        "BLOCK & REPORT": "బ్లాక్ చేసి రిపోర్ట్ చేయండి",
        "CAUTION": "జాగ్రత్త",
        "LIKELY SAFE": "బహుశా సురక్షితం",
    },
    "Kannada": {
        "BLOCK & REPORT": "ನಿರ್ಬಂಧಿಸಿ ಮತ್ತು ವರದಿ ಮಾಡಿ",
        "CAUTION": "ಎಚ್ಚರಿಕೆ",
        "LIKELY SAFE": "ಬಹುಶಃ ಸುರಕ್ಷಿತ",
    },
    "Malayalam": {
        "BLOCK & REPORT": "ബ്ലോക്ക് ചെയ്ത് റിപ്പോർട്ട് ചെയ്യുക",
        "CAUTION": "ജാഗ്രത",
        "LIKELY SAFE": "സുരക്ഷിതമായിരിക്കാം",
    },
}

ADVICE = {
    "BLOCK & REPORT": {
        "English": "This looks like a scam. Do not click any links, share OTPs or PINs, or send money. Block the sender and report it.",
        "Tamil": "இது மோசடி போல் தெரிகிறது. இணைப்புகளைக் கிளிக் செய்ய வேண்டாம், OTP அல்லது PIN பகிர வேண்டாம், பணம் அனுப்ப வேண்டாம். அனுப்பியவரைத் தடுத்து புகாரளிக்கவும்.",
        "Hindi": "यह एक धोखाधड़ी लगती है। किसी लिंक पर क्लिक न करें, OTP या PIN साझा न करें और पैसे न भेजें। भेजने वाले को ब्लॉक करके रिपोर्ट करें।",
        "Telugu": "ఇది మోసంలా కనిపిస్తోంది. లింకులపై క్లిక్ చేయకండి, OTP లేదా PIN షేర్ చేయకండి, డబ్బు పంపకండి. పంపినవారిని బ్లాక్ చేసి రిపోర్ట్ చేయండి.",
        "Kannada": "ಇದು ವಂಚನೆಯಂತೆ ಕಾಣುತ್ತಿದೆ. ಲಿಂಕ್‌ಗಳನ್ನು ಕ್ಲಿಕ್ ಮಾಡಬೇಡಿ, OTP ಅಥವಾ PIN ಹಂಚಿಕೊಳ್ಳಬೇಡಿ, ಹಣ ಕಳುಹಿಸಬೇಡಿ. ಕಳುಹಿಸಿದವರನ್ನು ಬ್ಲಾಕ್ ಮಾಡಿ ವರದಿ ಮಾಡಿ.",
        "Malayalam": "ഇത് തട്ടിപ്പായി തോന്നുന്നു. ലിങ്കുകളിൽ ക്ലിക്ക് ചെയ്യരുത്, OTP അല്ലെങ്കിൽ PIN പങ്കിടരുത്, പണം അയയ്ക്കരുത്. അയച്ചയാളെ ബ്ലോക്ക് ചെയ്ത് റിപ്പോർട്ട് ചെയ്യുക.",
    },
    "CAUTION": {
        "English": "This message is suspicious but not confirmed as a scam. Do not send money or share personal details until you verify the sender through a call or an official channel.",
        "Tamil": "இந்த செய்தி சந்தேகத்திற்குரியது, ஆனால் மோசடி என உறுதியாகவில்லை. அனுப்பியவரை அழைப்பு அல்லது அதிகாரப்பூர்வ வழியில் சரிபார்க்கும் வரை பணம் அனுப்பவோ தனிப்பட்ட விவரங்களைப் பகிரவோ வேண்டாம்.",
        "Hindi": "यह संदेश संदिग्ध है, लेकिन इसके धोखाधड़ी होने की पुष्टि नहीं है। भेजने वाले को कॉल या आधिकारिक माध्यम से सत्यापित करने तक पैसे न भेजें और निजी जानकारी साझा न करें।",
        "Telugu": "ఈ సందేశం అనుమానాస్పదంగా ఉంది, కానీ మోసమని నిర్ధారణ కాలేదు. పంపినవారిని కాల్ లేదా అధికారిక మార్గం ద్వారా నిర్ధారించుకునే వరకు డబ్బు పంపకండి, వ్యక్తిగత వివరాలు షేర్ చేయకండి.",
        "Kannada": "ಈ ಸಂದೇಶ ಅನುಮಾನಾಸ್ಪದವಾಗಿದೆ, ಆದರೆ ವಂಚನೆ ಎಂದು ದೃಢಪಟ್ಟಿಲ್ಲ. ಕಳುಹಿಸಿದವರನ್ನು ಕರೆ ಅಥವಾ ಅಧಿಕೃತ ಮಾರ್ಗದ ಮೂಲಕ ಪರಿಶೀಲಿಸುವವರೆಗೆ ಹಣ ಕಳುಹಿಸಬೇಡಿ, ವೈಯಕ್ತಿಕ ವಿವರಗಳನ್ನು ಹಂಚಿಕೊಳ್ಳಬೇಡಿ.",
        "Malayalam": "ഈ സന്ദേശം സംശയാസ്പദമാണ്, പക്ഷേ തട്ടിപ്പാണെന്ന് ഉറപ്പില്ല. അയച്ചയാളെ ഫോൺ വിളിച്ചോ ഔദ്യോഗിക മാർഗത്തിലൂടെയോ സ്ഥിരീകരിക്കുന്നതുവരെ പണം അയയ്ക്കരുത്, വ്യക്തിഗത വിവരങ്ങൾ പങ്കിടരുത്.",
    },
    "LIKELY SAFE": {
        "English": "This message looks normal and has no clear scam signs. Stay alert and never share OTPs or PINs.",
        "Tamil": "இந்த செய்தி சாதாரணமாகத் தெரிகிறது, மோசடி அறிகுறிகள் இல்லை. எச்சரிக்கையாக இருங்கள், OTP அல்லது PIN ஐ ஒருபோதும் பகிர வேண்டாம்.",
        "Hindi": "यह संदेश सामान्य लगता है और इसमें धोखाधड़ी के स्पष्ट संकेत नहीं हैं। सतर्क रहें और OTP या PIN कभी साझा न करें।",
        "Telugu": "ఈ సందేశం సాధారణంగా కనిపిస్తోంది, మోసం సంకేతాలు లేవు. అప్రమత్తంగా ఉండండి, OTP లేదా PIN ఎప్పుడూ షేర్ చేయకండి.",
        "Kannada": "ಈ ಸಂದೇಶ ಸಾಮಾನ್ಯವಾಗಿ ಕಾಣುತ್ತಿದೆ, ವಂಚನೆಯ ಸ್ಪಷ್ಟ ಸೂಚನೆಗಳಿಲ್ಲ. ಎಚ್ಚರಿಕೆಯಿಂದಿರಿ, OTP ಅಥವಾ PIN ಅನ್ನು ಎಂದಿಗೂ ಹಂಚಿಕೊಳ್ಳಬೇಡಿ.",
        "Malayalam": "ഈ സന്ദേശം സാധാരണമായി തോന്നുന്നു, തട്ടിപ്പിന്റെ വ്യക്തമായ സൂചനകളില്ല. ജാഗ്രത പാലിക്കുക, OTP അല്ലെങ്കിൽ PIN ഒരിക്കലും പങ്കിടരുത്.",
    },
}

FOLLOWUP_QUESTION = {
    "English": (
        "Can you share more details? For example, what the message or caller "
        "said, and whether it asked for money, an OTP or a link."
    ),
    "Tamil": (
        "மேலும் விவரங்களைப் பகிர முடியுமா? செய்தி அல்லது அழைப்பாளர் என்ன கூறினார், "
        "பணம், OTP அல்லது இணைப்பு கேட்டதா?"
    ),
    "Hindi": (
        "क्या आप और जानकारी दे सकते हैं? संदेश या कॉल करने वाले ने क्या कहा, "
        "और क्या उसने पैसे, OTP या लिंक मांगा?"
    ),
    "Telugu": (
        "దయచేసి మరిన్ని వివరాలు ఇవ్వగలరా? సందేశం లేదా కాలర్ ఏమి చెప్పారు, "
        "డబ్బు, OTP లేదా లింక్ అడిగారా?"
    ),
    "Kannada": (
        "ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರಗಳನ್ನು ನೀಡಬಹುದೇ? ಸಂದೇಶ ಅಥವಾ ಕರೆ ಮಾಡಿದವರು ಏನು ಹೇಳಿದರು, "
        "ಹಣ, OTP ಅಥವಾ ಲಿಂಕ್ ಕೇಳಿದರೇ?"
    ),
    "Malayalam": (
        "കൂടുതൽ വിവരങ്ങൾ പങ്കുവയ്ക്കാമോ? സന്ദേശത്തിലോ വിളിച്ചയാളോ എന്താണ് പറഞ്ഞത്, "
        "പണമോ OTP യോ ലിങ്കോ ചോദിച്ചോ?"
    ),
}


# ------------------------------------------------------------
# 3. PROMPTS (English only, short)
# ------------------------------------------------------------

CLASSIFY_PROMPT = """You are a scam detection assistant. Read the message and decide.

BLOCK & REPORT = clear fraud (OTP/PIN/password request, fake KYC, fake account
blocking, lottery or prize fee, fake refund, remote-access app, malicious link,
threats to force payment, phishing).
CAUTION = suspicious but not proven. Use it ONLY when the message asks for
money or personal details, or creates urgency, from an unverified sender
(for example a friend on a new number urgently asking for money).
LIKELY SAFE = normal message (delivery update, order update, bill payment
confirmation, bank transaction confirmation, reminders, ordinary chat). If the
message does not ask for money, OTP, links or personal details, choose
LIKELY SAFE, not CAUTION.

Database evidence:
{evidence}

Reply in exactly this format and nothing else:
VERDICT: <BLOCK & REPORT or CAUTION or LIKELY SAFE>
REASON: <one short sentence in English>"""

EXPLAIN_PROMPT = """You are a scam detection assistant. The message below was
flagged as a scam.

Evidence:
{evidence}

Write ONE short sentence in English explaining to the user why this message
is a scam. Do not repeat the word "evidence". Reply with the sentence only."""


# ------------------------------------------------------------
# 4. HELPERS
# ------------------------------------------------------------

# Words/links that can indicate a scam. If NONE of these appear and the
# database finds nothing, the message is treated as LIKELY SAFE without
# asking the LLM.
RISK_WORDS = re.compile(
    r"\b(otp|pin|cvv|password|passcode|kyc|aadhaar|aadhar|upi|anydesk|"
    r"teamviewer|quicksupport|apk|install|link|click|urgent|urgently|"
    r"immediately|blocked|suspended|expire|expired|verify|verification|"
    r"refund|prize|lottery|winner|won|fee|charges|send|transfer|pay|"
    r"deposit|loan|arrest|police|customs|gift card|investment|"
    r"new number|changed my number|account number|card number)\b"
    r"|https?://|www\.|bit\.ly|tinyurl",
    flags=re.IGNORECASE,
)

# Words that show a message is only CONFIRMING something that already happened.
CONFIRMATION_WORDS = re.compile(
    r"\b(successful|successfully|credited|debited|completed|processed|"
    r"delivered|received|confirmed|scheduled|paid|dispatched|shipped|"
    r"deposited)\b",
    flags=re.IGNORECASE,
)

# Words that ask the reader to DO something (typical of scams).
ACTION_REQUEST = re.compile(
    r"\b(share|send|click|tap|verify|update|enter|provide|reply|call|"
    r"install|download|submit|confirm your|login|log in|sign in|"
    r"pay now|pay a|claim|reactivate|unblock|otp|pin|cvv|password|"
    r"anydesk|teamviewer|quicksupport|urgent|urgently|immediately|"
    r"blocked|suspended|expire|arrest)\b",
    flags=re.IGNORECASE,
)

URL_PATTERN = re.compile(r"https?://|www\.|bit\.ly|tinyurl|\.in/|\.com/", flags=re.IGNORECASE)


def has_risk_signals(text: str) -> bool:
    return bool(RISK_WORDS.search(text))


def looks_like_plain_confirmation(text: str) -> bool:
    """
    True for messages that only confirm something that already happened
    (e.g. 'transfer successful', 'order delivered') and neither contain a
    link nor ask the reader to do anything.
    """
    return (
        bool(CONFIRMATION_WORDS.search(text))
        and not URL_PATTERN.search(text)
        and not ACTION_REQUEST.search(text)
    )


def build_evidence(pre: dict) -> tuple:
    """
    Returns (evidence_text, fallback_reason, fallback_verdict, strong).
    'strong' is True when the local database alone is enough to decide.
    """
    lines = []
    fallback_reason = ""
    fallback_verdict = "CAUTION"
    strong = False

    pm = pre.get("pattern_match", {})
    if pm.get("found"):
        top = pm["matches"][0]
        lines.append(
            f"- Matches known scam pattern: {top['category']} "
            f"(risk: {top['risk']}). {top['explanation']}"
        )
        fallback_reason = f"{top['category']}: {top['explanation']}"
        risk = str(top.get("risk", "")).lower()
        if ("high" in risk or "critical" in risk) and pm.get("confidence", 0) >= 0.5:
            fallback_verdict = "BLOCK & REPORT"
            strong = True
    else:
        lines.append("- No known scam pattern matched.")
        fallback_verdict = "LIKELY SAFE"
        fallback_reason = "No known scam patterns were found in this message."

    for key, res in pre.items():
        if key.startswith("upi:"):
            value = key[4:]
            if res.get("found"):
                lines.append(f"- UPI ID {value} is reported as a scam ({res['reports']} reports).")
                fallback_reason = f"The UPI ID {value} is reported as a scam."
                fallback_verdict = "BLOCK & REPORT"
                strong = True
            else:
                lines.append(f"- UPI ID {value} has no reports in the local database.")
        elif key.startswith("phone:"):
            value = key[6:]
            if res.get("found"):
                lines.append(f"- Phone number {value} is reported as a scam ({res['reports']} reports).")
                fallback_reason = f"The phone number {value} is reported as a scam."
                fallback_verdict = "BLOCK & REPORT"
                strong = True
            else:
                lines.append(f"- Phone number {value} has no reports in the local database.")

    return "\n".join(lines), fallback_reason, fallback_verdict, strong


def has_reported_id(pre: dict) -> bool:
    """True if any UPI ID or phone number in the message is a reported scam."""
    for key, res in pre.items():
        if (key.startswith("upi:") or key.startswith("phone:")) and res.get("found"):
            return True
    return False


def llm_call(prompt: str, user_text: str) -> str:
    """One short call to the local model."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        extra_body={"keep_alive": "30m"},
    )
    return (response.choices[0].message.content or "").strip()


def parse_classification(text: str):
    """Parse 'VERDICT: X / REASON: Y'. Returns (verdict, reason)."""
    verdict = None
    upper = text.upper()
    for v in VALID_VERDICTS:
        if v in upper:
            verdict = v
            break

    reason = ""
    m = re.search(r"REASON\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    return verdict, reason


def reason_is_bad(reason: str) -> bool:
    r = (reason or "").strip().lower()
    return (
        len(r) < 12
        or "verdict:" in r
        or "database" in r
        or "evidence" in r
        or r.startswith("[")
    )


def is_too_vague(text: str, pm: dict) -> bool:
    words = re.findall(r"\w+", text)
    return len(words) <= 4 and not pm.get("found")


# ------------------------------------------------------------
# 5. TRIAGE AGENT
# ------------------------------------------------------------

class LLMTriageAgent:

    def __init__(self):
        self.trace = []

    def _result(self, verdict, reason_en, confidence, language):
        """
        English: show the specific reason.
        Other languages: show the pre-translated advice, plus the English
        detail so nothing is lost.
        """
        labels = VERDICT_LABELS.get(language, {})
        if language == "English":
            reason = reason_en
            detail = ""
        else:
            reason = ADVICE[verdict].get(language, ADVICE[verdict]["English"])
            detail = reason_en

        return {
            "action": "verdict",
            "verdict": verdict,
            "verdict_label": labels.get(verdict, verdict),
            "reason": reason,
            "detail": detail,
            "confidence": confidence,
            "trace": self.trace,
        }

    def run(self, user_input: str, language: str = "English") -> dict:

        self.trace = []

        # ---- 1. instant offline checks ----
        pre = {"pattern_match": pattern_match(user_input)}
        for upi in extract_upi_ids(user_input):
            pre[f"upi:{upi}"] = check_upi_id(upi)
        for num in extract_phone_numbers(user_input):
            pre[f"phone:{num}"] = check_phone_number(num)

        self.trace.append({"step": "precheck:offline_db", "detail": pre})

        evidence, fb_reason, fb_verdict, strong = build_evidence(pre)

        # ---- 2. too vague: ask the user, no LLM needed ----
        if is_too_vague(user_input, pre["pattern_match"]):
            self.trace.append({"step": "decision:too_vague", "detail": {}})
            return {
                "action": "ask_user",
                "question": FOLLOWUP_QUESTION.get(language, FOLLOWUP_QUESTION["English"]),
                "trace": self.trace,
            }

        # ---- 3. plain confirmation (no link, no request): LIKELY SAFE ----
        if not has_reported_id(pre) and looks_like_plain_confirmation(user_input):
            self.trace.append({
                "step": "decision:plain_confirmation",
                "detail": {"verdict": "LIKELY SAFE"},
            })
            return self._result(
                "LIKELY SAFE",
                "This is a normal confirmation. It has no link and does not ask "
                "you to share anything or take any action.",
                0.85,
                language,
            )

        # ---- 4a. strong database evidence: verdict is instant ----
        if strong:
            self.trace.append({
                "step": "decision:database_verdict",
                "detail": {"verdict": "BLOCK & REPORT"},
            })
            reason = fb_reason
            # Only English needs a fresh LLM sentence; other languages use advice text.
            if language == "English":
                try:
                    text = llm_call(EXPLAIN_PROMPT.format(evidence=evidence), user_input)
                    if not reason_is_bad(text):
                        reason = text
                    self.trace.append({"step": "llm:explanation", "detail": text})
                except Exception as e:
                    self.trace.append({"step": "llm:skipped", "detail": str(e)})
            return self._result("BLOCK & REPORT", reason, 0.9, language)

        # ---- 4b. no scam signals at all: LIKELY SAFE, no LLM needed ----
        pm_found = pre["pattern_match"].get("found", False)
        if not pm_found and not has_risk_signals(user_input):
            self.trace.append({
                "step": "decision:no_risk_signals",
                "detail": {"verdict": "LIKELY SAFE"},
            })
            return self._result(
                "LIKELY SAFE",
                "This looks like a normal message. It does not ask for money, "
                "OTPs, links or personal details.",
                0.85,
                language,
            )

        # ---- 4c. unclear: the local LLM decides ----
        try:
            text = llm_call(CLASSIFY_PROMPT.format(evidence=evidence), user_input)
            self.trace.append({"step": "llm:classify", "detail": text})

            verdict, reason = parse_classification(text)
            if verdict is None:
                verdict = fb_verdict
            if reason_is_bad(reason):
                reason = fb_reason or (
                    "Please verify the sender through an official channel "
                    "before taking any action."
                )
            return self._result(verdict, reason, 0.65, language)

        except Exception as e:
            self.trace.append({"step": "llm:error_fallback", "detail": str(e)})
            return self._result(
                fb_verdict,
                fb_reason or "Please verify the sender before taking any action.",
                0.5,
                language,
            )


# ------------------------------------------------------------
# 6. LOCAL TEST
# ------------------------------------------------------------

if __name__ == "__main__":
    agent = LLMTriageAgent()
    for msg in [
        "Dear customer your account will be blocked. Update your KYC now: http://bit.ly/xyz123",
        "BANK ALERT: Your scheduled monthly transfer of $250.00 to savings was successful on 09/30/2026. Ref: #99421",
        "Your order #48213 has been delivered successfully. Thank you for shopping with us.",
    ]:
        r = agent.run(msg)
        print(r["verdict"], "-", r["reason"])