import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.jsonl"

# Local embedding model (free, offline)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    os.makedirs("index", exist_ok=True)

    print("Loading local embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks()
    texts = [f"{c['title']}\n{c['content']}" for c in chunks]

    print("Creating local embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Index built successfully (LOCAL embeddings).")
    print(f"Vectors indexed: {len(chunks)}")

if __name__ == "__main__":
    main()
