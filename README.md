# Nexus RAG Intelligence
**Agentic Multi-Source Data Retrieval & Synthesis Engine**

An enterprise-grade, multi-document Retrieval-Augmented Generation (RAG) system. This project implements an Agentic Router to dynamically classify user intent, utilize mathematically normalized Hybrid Search (Dense + Sparse), and generate highly structured professional reports via a custom Gradio web interface.

## 🎥 System Demo

https://github.com/user-attachments/assets/abd1a3ed-40b9-4645-a48b-669a7eecfb23

---

## Problem Statement
Standard RAG pipelines often struggle when faced with diverse, multi-format datasets (PDFs, CSVs, Markdown) and complex user queries. Traditional systems typically use a "one-size-fits-all" retrieval strategy, leading to poor performance when users ask for broad summaries versus highly specific data correlations. Furthermore, combining keyword search (BM25) with semantic search (VectorDB) often results in mismatched scoring scales, making it difficult to accurately measure confidence. 

## 💡 The Solution
Nexus RAG solves these challenges by implementing an **Agentic Dispatch Architecture**:
1. **Semantic Routing:** An LLM-based router evaluates the user's query and classifies the intent into `FACTUAL`, `COMPARATIVE`, or `SUMMARY`.
2. **Dynamic K-Retrieval:** The system dynamically adjusts the number of retrieved chunks based on the intent (e.g., pulling 12 chunks for deep comparative analysis, but only 6 for quick factual extraction).
3. **Normalized Hybrid Search:** It retrieves documents using both ChromaDB (Semantic) and Rank_BM25 (Keyword), mathematically normalizing their vastly different scoring systems (L2 Distance vs. BM25 Frequency) onto a standard 0-100% confidence scale.
4. **Professional UI & State Management:** A custom CSS-styled Gradio interface that maintains isolated session states per user, preventing data bleed while offering dynamic routing diagnostics.

---

## ✨ Key Features

* **Multi-Format Ingestion:** Seamlessly processes `.pdf`, `.csv`, and `.md` files, automatically injecting source-file metadata for accurate citations.
* **LLM-Agnostic Setup:** Built using LangChain and OpenRouter, currently defaulting to `nvidia/nemotron-3-nano-30b` but easily swappable to Claude, GPT-4, or Llama.
* **Retriever-Level Citations:** Eliminates LLM hallucination in citations by directly mapping the retrieved VectorDB chunks to the final UI output.
* **Enterprise UI:** A responsive, dark-mode compatible Gradio interface featuring floating cards, a hidden diagnostic accordion, and rich Markdown table rendering.

---

## 🛠️ Technology Stack
* **Framework:** LangChain (`langchain-community`, `langchain-openai`)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Sparse Retrieval:** Rank_BM25
* **Frontend:** Gradio 
* **Data Processing:** Pandas, Unstructured, PyPDF

