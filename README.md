# Local RAG Pipeline (Retrieval-Augmented Generation)

A simple local RAG (Retrieval-Augmented Generation) system built in Python that allows you to ask questions over your own documents using embeddings and a local LLM.

---

# Project Overview

This project demonstrates a full RAG pipeline:

User Query  
→ Load Documents  
→ Split into Chunks  
→ Create Embeddings  
→ Store in Vector Database  
→ Retrieve Relevant Chunks  
→ Generate Answer using Local LLM  



# Key Features

- Fully local execution (no need for paid APIs)
- Document ingestion and chunking
- Embedding generation for semantic search
- Vector-based retrieval system
- Context-aware response generation
- Beginner-friendly GenAI architecture

---

# Tech Stack

- Python
- LangChain (or custom pipeline if not used)
- ChromaDB / FAISS (vector store)
- Ollama / Local LLM (e.g., Llama3, Mistral)
- Sentence Transformers (embeddings)


