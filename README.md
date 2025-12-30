# Resume Chatbot — Local RAG System

A fully local, resume-aware AI chatbot that allows recruiters to ask natural-language questions about my experience and receive **grounded, first-person conversational responses** based strictly on my resume.

This project is designed as a **demonstration of Retrieval-Augmented Generation (RAG)**, semantic search, and practical LLM integration, with a strong emphasis on **grounding, privacy, and system design tradeoffs**.

---

##  Key Features

- Resume ingestion and structured chunking
- Semantic retrieval using transformer embeddings + FAISS
- Local LLM reasoning via Ollama
- First-person, conversational answers (not resume copy-paste)
- Strict grounding so that the model cannot hallucinate
