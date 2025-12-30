# **💬 Resume-Chatbot (Local RAG System)**

A fully local, AI-powered resume chatbot designed to allow recruiters and hiring managers to ask natural-language questions about my background and receive **grounded, first-person conversational responses**, built to demonstrate real-world applications of semantic search, Retrieval-Augmented Generation (RAG), and LLM system design.

Tech Stack: Python • Sentence-Transformers • FAISS • Ollama (Mistral) • Streamlit

**Key Contributions & Impact**

- Designed and implemented a full Retrieval-Augmented Generation (RAG) pipeline that ingests a resume PDF, chunks and embeds content, and retrieves semantically relevant sections in response to recruiter-style questions.

- Built a local semantic search layer using transformer embeddings and FAISS, enabling accurate retrieval of experience, research, projects, and skills without keyword matching.

- Integrated a local instruction-tuned LLM (Mistral via Ollama) to generate concise, first-person explanations that sound like a real candidate conversation rather than resume bullet points.

- Enforced strict grounding rules to prevent hallucinations, to ensure all responses are derived exclusively from retrieved resume content.

- Developed an interactive Streamlit chat interface that simulates a recruiter Q&A experience, allowing users to explore background, projects, and skills through natural dialogue.

- Optimized the system for low-memory environments by selecting lightweight models and limiting context size, demonstrating practical engineering tradeoffs between performance, stability, and answer quality.

**System Capabilities**

- Recruiter Q&A: Supports questions such as “What did you do during your internship?” or “Tell me about your research experience.”

- Grounded Responses: Answers are strictly based on resume content, with explicit acknowledgment when information is not present.

- Conversational Tone: Outputs are concise, professional, and framed in first person to mirror real interview responses.

- Privacy-First Design: Fully local execution which means no cloud APIs were used and no resume data uploaded externally.

- Scalable Architecture: Modular design allows easy extension to role-specific modes (Data Science, Analytics, Finance) or cloud deployment if desired.
