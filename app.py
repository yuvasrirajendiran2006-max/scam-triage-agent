import os
import tempfile

import streamlit as st

from llm_agent import LLMTriageAgent
from agent import TriageAgent
from transcribe import transcribe

st.set_page_config(page_title="Scam Triage Agent", page_icon="🛡️", layout="centered")

st.title("🛡️ Scam / Fraud Triage Agent")
st.caption(
    "Runs fully on-device. Paste a suspicious SMS, call transcript, or UPI request, "
    "or upload a call recording. Nothing leaves your laptop."
)

LANG_OPTIONS = ["English", "Tamil", "Hindi", "Telugu", "Kannada", "Malayalam"]

# ---------------- session state ----------------
if "agent" not in st.session_state:
    st.session_state.agent = LLMTriageAgent()
if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False
if "original_input" not in st.session_state:
    st.session_state.original_input = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "language" not in st.session_state:
    st.session_state.language = "English"
if "transcripts" not in st.session_state:
    st.session_state.transcripts = {}

if st.button("🔄 New Investigation"):
    st.session_state.awaiting_followup = False
    st.session_state.original_input = ""
    st.session_state.pending_question = ""
    st.session_state.user_input_box = ""
    st.session_state.followup_box = ""
    st.session_state.agent = LLMTriageAgent()
    st.rerun()


# ---------------- rendering ----------------
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
        st.warning(f"🤔 {result['question']}")
        st.session_state.awaiting_followup = True
        st.session_state.pending_question = result["question"]
    else:
        color = {
            "BLOCK & REPORT": "red",
            "CAUTION": "orange",
            "LIKELY SAFE": "green",
        }.get(result["verdict"], "gray")
        label = result.get("verdict_label", result["verdict"])
        st.markdown(f"### Verdict: :{color}[{label}]")
        if label != result["verdict"]:
            st.caption(result["verdict"])
        st.write(result["reason"])
        if result.get("detail"):
            st.caption(f"Details (English): {result['detail']}")
        st.progress(result["confidence"])
        st.session_state.awaiting_followup = False


def get_transcript(uploaded):
    """Transcribe once per uploaded file and cache it."""
    key = f"{uploaded.name}-{uploaded.size}"
    if key not in st.session_state.transcripts:
        suffix = os.path.splitext(uploaded.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            path = tmp.name
        try:
            with st.spinner("Transcribing on-device..."):
                st.session_state.transcripts[key] = transcribe(path)
        finally:
            os.remove(path)
    return st.session_state.transcripts[key]


# ---------------- main flow ----------------
if not st.session_state.awaiting_followup:

    language = st.selectbox(
        "Reply language",
        LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.language),
    )
    st.session_state.language = language

    audio = st.file_uploader(
        "Or upload a call recording", type=["wav", "mp3", "m4a"]
    )
    transcript = ""
    if audio is not None:
        try:
            transcript = get_transcript(audio)
            st.text_area("Transcript", transcript, height=100, disabled=True)
        except Exception as e:
            st.error(f"Transcription failed: {e}")

    user_input = st.text_area(
        "Paste the suspicious message / transcript / UPI ID here:",
        height=150,
        key="user_input_box",
    )

    text_to_check = user_input.strip() or transcript.strip()

    if st.button("Investigate", type="primary") and text_to_check:
        st.session_state.original_input = text_to_check
        try:
            with st.spinner("Investigating on-device..."):
                result = st.session_state.agent.run(text_to_check, language=language)
        except Exception as e:
            st.error(f"Local LLM error: {e}. Falling back to rule-based agent.")
            result = TriageAgent().run(text_to_check)
        render_result(result)

else:
    st.info(f"Follow-up: {st.session_state.pending_question}")
    followup = st.text_area("Your answer:", height=80, key="followup_box")

    if st.button("Submit answer", type="primary") and followup.strip():
        combined = f"{st.session_state.original_input}\n{followup}"
        try:
            with st.spinner("Investigating on-device..."):
                result = st.session_state.agent.run(
                    combined, language=st.session_state.language
                )
        except Exception as e:
            st.error(f"Local LLM error: {e}. Falling back to rule-based agent.")
            result = TriageAgent().run(combined)
        render_result(result)

st.divider()
st.caption(
    "Try: 'Dear customer your account will be blocked. Update your KYC now: "
    "http://bit.ly/xyz123' or 'I got a message that says verify to receive refund'"
)