# On-Device Scam / Fraud Triage Agent

Snapdragon AI Lab Build & Present Challenge submission.

Paste a suspicious SMS, call transcript or UPI request, or upload a call
recording. The agent gives a clear verdict (**BLOCK & REPORT**, **CAUTION** or
**LIKELY SAFE**) with advice in English, Tamil, Hindi, Telugu, Kannada or
Malayalam.

Everything runs on the laptop. No message, call or audio is sent to the cloud.

**Demo video:** PASTE_YOUR_VIDEO_LINK_HERE

## The problem

Scam SMS, fake KYC alerts, fake refund requests and scam calls are common in
India. Checking a message with a cloud AI means uploading private texts, bank
details and call audio to someone else's server. Many people also have weak or
no internet.

## Why on-device

- **Private:** messages, UPI IDs and call recordings never leave the laptop
- **Offline:** works without internet after setup
- **Fast:** the common cases are answered instantly from a local database, with no network round trip

## What it does

1. Paste text or upload a call recording (WAV, MP3, M4A)
2. Call audio is turned into text on-device (Whisper)
3. The message is checked against a local scam-pattern database and a local list of reported UPI IDs and phone numbers
4. A small local LLM explains the verdict or judges unclear cases
5. The result is shown in the user's chosen language

## How the decision works

| Situation | What happens |
|---|---|
| Too vague ("Someone contacted me") | Asks a follow-up question |
| Plain confirmation with no link and no request (order delivered, transfer successful) | LIKELY SAFE, instantly |
| Strong local database match or reported UPI ID / number | BLOCK & REPORT, instantly |
| No scam signals at all | LIKELY SAFE, instantly |
| Anything unclear | The local LLM decides in one short call |

The LLM is used only where judgment is needed. This keeps the agent fast on
a laptop and reduces wrong answers from a small model.

## Models used (all open-source, run locally)

- **Llama 3.2 1B** via Ollama: explanations and unclear cases
- **Whisper (base)** via faster-whisper: speech-to-text for call recordings

## Languages

English, Tamil, Hindi, Telugu, Kannada and Malayalam. Verdict labels and
advice are pre-written in each language so they appear instantly. The
LLM's specific explanation is generated in English and shown below the advice.

## Run it

Requirements: Python 3.10+, [Ollama](https://ollama.com)

```
pip install -r requirements.txt
ollama pull llama3.2:1b
streamlit run app.py
```

Whisper downloads its model the first time you upload audio (internet needed
once). After that, everything works offline.

## Project files

- `app.py`: Streamlit interface
- `llm_agent.py`: triage logic and local LLM calls
- `tools.py`: local scam-pattern and reported-ID checks
- `transcribe.py`: on-device speech-to-text
- `agent.py`: rule-based fallback if the local LLM is unavailable
- `data/`: `scam_patterns.json`, `reported_ids.json`

## Limitations

- The scam database is small and hand-built; a real deployment needs a larger, regularly updated one
- A 1B model can misjudge edge cases, so unclear results lean on the local rules
- Tamil and other language advice is pre-written, not generated live
- Currently runs on CPU through Ollama

## Snapdragon roadmap

Built for Snapdragon-powered laptops, where private on-device AI matters most.
Next steps: move the LLM and Whisper to Qualcomm AI Hub models for NPU
acceleration, and add live translation with an on-device multilingual model.

## Author

Yuvasri, PSG Institute of Technology and Applied Research
