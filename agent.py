"""
Scam/Fraud Triage Agent — core decision loop.

Design: observe -> decide -> act -> evaluate -> adapt -> verdict
"""
from tools import pattern_match, extract_upi_ids, extract_phone_numbers, check_upi_id, check_phone_number, request_more_info


class TriageAgent:
    def __init__(self, planner="heuristic"):
        self.planner = planner
        self.trace = []  # every step gets logged here for the UI / demo

    def log(self, step, detail):
        entry = {"step": step, "detail": detail}
        self.trace.append(entry)
        return entry

    def run(self, user_input: str, follow_up_answer: str = None) -> dict:
        """
        Run one full triage cycle. If the agent needs more info, it returns
        an 'ask_user' action instead of a verdict — the caller (UI/CLI) should
        collect the answer and call run() again, this time combining the
        original input with follow_up_answer.
        """
        self.trace = []
        combined_text = user_input if not follow_up_answer else f"{user_input}\n{follow_up_answer}"

        # 1. OBSERVE
        self.log("observe", f"Received input: {combined_text[:200]}")

        # 2. DECIDE + ACT: always start with pattern matching — cheapest signal
        pm_result = pattern_match(combined_text)
        self.log("act:pattern_match", pm_result)

        # 3. EVALUATE: is pattern confidence high enough to decide?
        if pm_result["found"] and pm_result["confidence"] >= 0.6:
            top = pm_result["matches"][0]
            return self._verdict(
                verdict="BLOCK & REPORT",
                reason=f"Strong match to known scam pattern: {top['category']}. {top['explanation']}",
                confidence=top["confidence"],
            )

        # 4. ADAPT: pattern match was inconclusive or absent — try ID/number lookups
        self.log("evaluate", "Pattern match inconclusive/absent — adapting: checking for UPI IDs / phone numbers")

        upi_ids = extract_upi_ids(combined_text)
        phones = extract_phone_numbers(combined_text)

        for uid in upi_ids:
            result = check_upi_id(uid)
            self.log(f"act:check_upi_id({uid})", result)
            if result["found"] and result["reports"] >= 5:
                return self._verdict(
                    verdict="BLOCK & REPORT",
                    reason=f"UPI ID '{uid}' has {result['reports']} prior scam reports. {result['note']}",
                    confidence=0.9,
                )

        for num in phones:
            result = check_phone_number(num)
            self.log(f"act:check_phone_number({num})", result)
            if result["found"] and result["reports"] >= 5:
                return self._verdict(
                    verdict="BLOCK & REPORT",
                    reason=f"Number '{num}' has {result['reports']} prior scam reports. {result['note']}",
                    confidence=0.9,
                )

        # 5. Still ambiguous — weak pattern match, but nothing conclusive
        if pm_result["found"] and pm_result["confidence"] >= 0.3:
            top = pm_result["matches"][0]
            if follow_up_answer is None:
                q = request_more_info(
                    f"This partially resembles a '{top['category']}' scam pattern but I'm not fully "
                    f"confident. Did the message ask you to click a link, share an OTP/PIN, or make a "
                    f"payment? Can you share the exact link or UPI ID involved?"
                )
                self.log("adapt:request_more_info", q)
                return {"action": "ask_user", "question": q["question"], "trace": self.trace}
            else:
                return self._verdict(
                    verdict="CAUTION",
                    reason=f"Weak resemblance to '{top['category']}' pattern, and follow-up answer did not "
                           f"add strong confirming or disconfirming evidence. Recommend caution: do not click "
                           f"links, share OTP/PIN, or make payments until verified through official channels.",
                    confidence=top["confidence"],
                )

        # 6. No signal at all
        return self._verdict(
            verdict="LIKELY SAFE",
            reason="No known scam patterns, reported UPI IDs, or reported numbers matched. "
                   "Still exercise normal caution with unsolicited requests for money or personal info.",
            confidence=0.4,
        )

    def _verdict(self, verdict, reason, confidence):
        self.log("verdict", {"verdict": verdict, "reason": reason, "confidence": confidence})
        return {
            "action": "verdict",
            "verdict": verdict,
            "reason": reason,
            "confidence": confidence,
            "trace": self.trace,
        }


if __name__ == "__main__":
    agent = TriageAgent()

    print("=== Test 1: Clear scam ===")
    result = agent.run("Dear customer your account will be blocked. Update your KYC now: http://bit.ly/xyz123")
    print(result["verdict"], "-", result["reason"])
    print()

    print("=== Test 2: Ambiguous case (adaptation test) ===")
    ambiguous_msg = "Hi, I got a message that says verify to receive refund, does this look ok?"
    result = agent.run(ambiguous_msg)
    if result["action"] == "ask_user":
        print("Agent asks:", result["question"])
        result2 = agent.run(
            ambiguous_msg,
            follow_up_answer="It asked me to enter my UPI PIN to receive the cashback."
        )
        print(result2["verdict"], "-", result2["reason"])
    else:
        print(result["verdict"], "-", result["reason"])