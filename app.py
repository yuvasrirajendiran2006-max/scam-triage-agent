import streamlit as st
from agent import TriageAgent

st.set_page_config(page_title="Scam Triage Agent", page_icon="🛡️", layout="centered")

st.title("🛡️ Scam / Fraud Triage Agent")
st.caption("Paste a suspicious SMS, call transcript, or UPI request. The agent will investigate step-by-step.")

if "agent" not in st.session_state:
    st.session_state.agent = TriageAgent()
if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False
if "original_input" not in st.session_state:
    st.session_state.original_input = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if st.button("🔄 New Investigation"):
    st.session_state.awaiting_followup = False
    st.session_state.original_input = ""
    st.session_state.pending_question = ""
    st.session_state.user_input_box = ""
    st.session_state.followup_box = ""
    st.rerun()

def render_trace(trace):
    with st.expander("🔍 Agent reasoning trace", expanded=False):
        for step in trace:
            st.markdown(f"**{step['step']}**")
            if isinstance(step["detail"], dict):
              st.json(step["detail"])
            else:
              st.write(step["detail"])

def render_result(result):
    render_trace(result["trace"])
    if result["action"] == "ask_user":
        st.warning(f"🤔 Agent needs more info: {result['question']}")
        st.session_state.awaiting_followup = True
        st.session_state.pending_question = result["question"]
    else:
        color = {"BLOCK & REPORT": "red", "CAUTION": "orange", "LIKELY SAFE": "green"}.get(result["verdict"], "gray")
        st.markdown(f"### Verdict: :{color}[{result['verdict']}]")
        st.write(result["reason"])
        st.progress(result["confidence"])
        st.session_state.awaiting_followup = False

if not st.session_state.awaiting_followup:
    user_input = st.text_area("Paste the suspicious message / transcript / UPI ID here:", height=150, key="user_input_box")
    if st.button("Investigate", type="primary") and user_input.strip():
        st.session_state.original_input = user_input
        result = st.session_state.agent.run(user_input)
        render_result(result)
else:
    st.info(f"Follow-up: {st.session_state.pending_question}")
    followup = st.text_area("Your answer:", height=80, key="followup_box")
    if st.button("Submit answer", type="primary") and followup.strip():
        result = st.session_state.agent.run(st.session_state.original_input, follow_up_answer=followup)
        render_result(result)

st.divider()
st.caption("Try: 'Dear customer your account will be blocked. Update your KYC now: http://bit.ly/xyz123' "
           "or 'I got a message that says verify to receive refund'")