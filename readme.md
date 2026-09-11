# Autonomous Multi-Agent Document Research & QA System

An end-to-end, zero-cost, enterprise-grade Retrieval-Augmented Generation (RAG) platform powered by an autonomous multi-agent state machine. The system ingests dense, domain-specific technical documentation (PDFs/Text), chunks and vectorizes embeddings locally, and orchestrates specialized LLM agents (Retriever, Critic, Synthesizer) to produce grounded, hallucination-resistant answers with verifiable source attribution.

---

## 🔗 Live Application & Demo

* **Live Interactive Demo:** [Launch on Streamlit Community Cloud](https://www.google.com/search?q=https://your-streamlit-app-link-here.streamlit.app/)
---

## 📸 System Architecture & Interface

### Application Interface


*Figure 1: Streamlit-based interactive workbench featuring file ingestion, dynamic retrieval confidence, real-time agent execution telemetry, and streaming synthesized answers.*
![Multi-Agent UI](https://raw.githubusercontent.com/your-username/multi_agent_rag/main/assets/ui_overview.png)

### Multi-Agent Autonomous State Machine

```text
                  +-------------------------+
                  |    User Query Input     |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  |     Retriever Agent     |
                  | (Top-k Similarity Search|
                  |     via ChromaDB)       |
                  +------------+------------+
                               |
                   [Retrieved Context Blocks]
                               |
                               v
                  +-------------------------+
                  |       Critic Agent      |
                  |  (Deterministic JSON    |
                  |   Hallucination Audit)  |
                  +------------+------------+
                               |
              +----------------+----------------+
              |                                 |
      [Status: Grounded]               [Status: Ambiguous / Insufficient]
              |                                 |
              v                                 v
   +-----------------------+        +-----------------------+
   |   Synthesizer Agent   |        |   Rejection / Clarify |
   |  (Streaming Markdown  |        |  (Safe Fallback Loop) |
   |   with Citations)     |        +-----------------------+
   +-----------+-----------+
               |
               v
     [Render Output to UI]

```

*Figure 2: Deterministic routing pipeline enforcing zero hallucination via automated cross-validation.*
![Multi-Agent UI](https://raw.githubusercontent.com/your-username/multi_agent_rag/main/assets/ui_overview.png)~~~~~~~~~~~~

---

## 💡 Key Architectural Features

* **Zero-Cost Production Stack:** 100% perpetual free-tier operational model utilizing Hugging Face local embeddings, ephemeral in-memory/on-disk vector storage, and Google GenAI API endpoints.
* **Autonomous Multi-Agent Orchestration:**
* **Retriever Agent:** Computes dense vector embeddings and executes semantic similarity searches against document chunks using localized vector indexing.
* **Critic Agent:** Operates deterministically at low temperature (`0.1`) with structured JSON schema enforcement (`application/json`) to audit retrieved context relevance before synthesis, preventing hallucinations.
* **Synthesizer Agent:** Transforms verified document fragments into cohesive, citation-backed markdown responses with full streaming support (`generate_content_stream`).


* **Deterministic Chunking & Token Management:** Implements text splitting optimized for sentence boundaries, preserving semantic fidelity for dense domain materials (e.g., technical equations, formulas, state tables).
* **Defensive Error Handling:** Built-in fail-safe states for missing context, token overflows, corrupted payloads, and API disconnects.

---

## 🛠️ Technology Stack

| Layer | Component | Description / Rationalization |
| --- | --- | --- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Lightweight, state-driven UI framework with native streaming support. |
| **LLM Inference** | Unified Google GenAI SDK | Free tier inference for agentic reasoning and high-speed streaming. |
| **Embedding Engine** | `sentence-transformers` | Open-source Hugging Face dense vector model runs locally with zero API cost. |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Serverless, embeddable vector store supporting fast cosine similarity search. |
| **Document Parsing** | `PyPDF2` / Python Native | Robust parsing pipeline for text extraction and segment normalization. |

---

## 📂 Directory Structure

```text
multi_agent_rag/
├── .streamlit/
│   ├── config.toml           # UI styling and server configurations
│   └── secrets.toml          # Encrypted local API keys (Git ignored)
├── assets/
│   ├── ui_overview.png       # UI demonstration image
│   └── architecture.png      # High-level architecture diagram
├── src/
│   ├── __init__.py
│   ├── config.py             # Global constants, paths, and logger setup
│   ├── document_processor.py # PDF/text extraction, chunking, and normalization
│   ├── vector_store.py       # ChromaDB lifecycle, embedding index, and queries
│   └── agents/
│       ├── __init__.py
│       ├── gemini_client.py  # Unified GenAI client (Critic & Synthesizer)
│       └── workflow.py       # Agentic state machine router
├── .gitignore
├── LICENSE
├── README.md
├── app.py                    # Main Streamlit orchestration and session state
└── requirements.txt          # Python dependencies

```

---

## 🚀 Local Development Setup

### 1. Prerequisites

* Python 3.10, 3.11, or 3.12
* Git
* A free Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/multi_agent_rag.git
cd multi_agent_rag

# Create an isolated virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Secrets Configuration

Create a local secrets file inside `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

```

*(Ensure `.streamlit/secrets.toml` is present in your `.gitignore` to prevent secret leaks).*

### 4. Run the Application

```bash
streamlit run app.py

```

Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push your completed codebase to a public or private GitHub repository.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and authenticate with your GitHub account.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Expand **Advanced Settings** -> **Secrets** and inject your production API key:
```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

```


5. Click **Deploy**. Your app will automatically build, install packages from `requirements.txt`, and deploy on a public URL.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.