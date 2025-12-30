import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# PATHS & CONFIG
INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# LOADERS
def load_metadata():
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_model_and_index():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    metadata = load_metadata()
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
- Explain experiences naturally, not like a resume.
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
            "The local language model encountered an internal error while generating the response. "
            "Please try again."
        )

    except requests.exceptions.RequestException:
        return (
            "Unable to reach the local language model. "
            "Please ensure Ollama is running."
        )


# MAIN LOOP
def main():
    print("Resume Chatbot (Local, Conversational)")
    print("Type 'exit' to quit.\n")

    model, index, metadata = load_model_and_index()

    while True:
        question = input("Ask a recruiter-style question: ")
        if question.lower() == "exit":
            break

        answer = generate_answer(question, model, index, metadata)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
