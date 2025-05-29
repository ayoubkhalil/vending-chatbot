import streamlit as st
import requests

# --- Config ---
st.set_page_config(page_title="Vending Machine Chatbot", layout="centered")
HF_TOKEN = "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"  # Replace this securely

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Models ---
QWEN_MODEL = "ayoub-66/qwen-1.5b-error-classification-hff"
MBART_MODEL = "ayoub-66/mbart-vending-error-model"

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "turn_counter" not in st.session_state:
    st.session_state.turn_counter = 0
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None


# --- Functions ---

def ask_qwen_model(user_input):
    """Send message to Qwen model"""
    history = "\n".join([f"Customer: {msg}" for msg in st.session_state.chat_history])
    prompt = f"""You are a helpful vending machine support assistant.
Only output one natural follow-up question to clarify the user's problem.

{history}
Customer: {user_input}
Assistant:"""

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{QWEN_MODEL}",
        headers=HEADERS,
        json={"inputs": prompt}
    )
    result = response.json()
    return result[0]["generated_text"].replace(prompt, "").strip() if isinstance(result, list) else str(result)


def ask_mbart_model(convo_text):
    """Send full conversation to MBART model"""
    prompt = f"""
You are an AI assistant for vending machine diagnostics.

Based on the full conversation, extract:
- ErrorMessage
- ErrorCause
- ErrorBau
- ErrorKindTypeKey
- ErrorKindTypeName
- ErrorCauseTypeName
- ErrorCauseTypeKey
- RecommendedTechnician

Respond ONLY in this JSON format:
{{
  "ErrorMessage": "...",
  "ErrorCause": "...",
  "ErrorBau": "...",
  "ErrorKindTypeKey": "...",
  "ErrorKindTypeName": "...",
  "ErrorCauseTypeName": "...",
  "ErrorCauseTypeKey": "...",
  "RecommendedTechnician": "..."
}}

Conversation:
{convo_text}
"""
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MBART_MODEL}",
        headers=HEADERS,
        json={"inputs": prompt}
    )
    return response.json()


# --- UI ---

st.title("🤖 Vending Machine Diagnostic Chatbot")
st.markdown("Ask in English or German. After 3 messages, a diagnosis is generated.")

user_input = st.text_input("Your message:", key="user_input")

if st.button("Send") and user_input.strip():
    st.session_state.chat_history.append(user_input)
    st.session_state.turn_counter += 1

    if st.session_state.turn_counter < 3:
        reply = ask_qwen_model(user_input)
        st.write(f"🧠 Assistant: {reply}")
    elif st.session_state.diagnosis is None:
        full_convo = "\n".join([f"Customer: {m}" for m in st.session_state.chat_history])
        st.session_state.diagnosis = ask_mbart_model(full_convo)
        st.write("✅ Diagnosis:")
        st.code(st.session_state.diagnosis, language='json')
    else:
        st.write("🔁 Diagnosis already provided. Refresh to restart.")

if st.button("🔄 Reset Session"):
    st.session_state.chat_history = []
    st.session_state.turn_counter = 0
    st.session_state.diagnosis = None
    st.experimental_rerun()
