import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_metadata():
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def main():
    print("Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    metadata = load_metadata()

    while True:
        question = input("\nAsk a recruiter-style question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        q_vec = model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)

        scores, idxs = index.search(q_vec, k=3)

        print("\nTop relevant resume sections:\n")
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
            item = metadata[idx]
            print(f"{rank}. {item['title']}")
            print(f"   Type: {item['type']}")
            print(f"   Score: {score:.3f}")
            print(f"   Preview: {item['content'][:200]}...")
            print("-" * 60)

if __name__ == "__main__":
    main()
