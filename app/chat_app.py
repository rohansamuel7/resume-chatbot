import streamlit as st
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# CONFIG
INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# LOADERS
@st.cache_resource
def load_model_index_metadata():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)

    metadata = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    return model, index, metadata


# RETRIEVAL
def retrieve_chunks(question, model, index, metadata, k=2):
    q_vec = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)

    scores, idxs = index.search(q_vec, k)
    return [metadata[i] for i in idxs[0]]


# GENERATION
def generate_answer(question, model, index, metadata):
    chunks = retrieve_chunks(question, model, index, metadata, k=2)

    context = "\n\n".join(
        f"{c['title']}:\n{c['content']}" for c in chunks
    )

    prompt = f"""
You are the candidate speaking to a recruiter.

Rules:
- Speak in first person ("I", "my").
- Be concise, professional, and conversational.
- Explain experiences naturally.
- Use ONLY the information in the CONTEXT.
- If something is not explicitly stated, say so honestly.
- Focus on learning, impact, and relevance.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "num_predict": 150,
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            },
            timeout=300
        )

        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            return "I could not generate a response based on the provided resume context."

        return answer

    except requests.exceptions.HTTPError:
        return (
            "The local language model encountered an internal error. "
            "Please try asking again."
        )

    except requests.exceptions.RequestException:
        return (
            "Unable to connect to the local language model. "
            "Please ensure Ollama is running."
        )


# STREAMLIT UI 
st.set_page_config(page_title="Resume Chatbot", layout="centered")

st.title("💬 Resume Chatbot")
st.caption("Ask recruiter-style questions. Answers are grounded in my resume.")

model, index, metadata = load_model_index_metadata()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar suggested questions
with st.sidebar:
    st.subheader("Suggested Questions")
    suggestions = [
        "What did you do during your internship?",
        "Tell me about your research at Penn State",
        "What was your Tesla project about?",
        "What skills are you strongest in?",
        "What roles are you best suited for?"
    ]

    for q in suggestions:
        if st.button(q):
            st.session_state.chat_history.append(("user", q))
            answer = generate_answer(q, model, index, metadata)
            st.session_state.chat_history.append(("assistant", answer))

# Chat input
user_input = st.chat_input("Ask a recruiter-style question...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    answer = generate_answer(user_input, model, index, metadata)
    st.session_state.chat_history.append(("assistant", answer))

# Render chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
