# 🔬 RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.oracle.com/badge/Streamlit-1.42+-red.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini--2.5--Flash-blueviolet.svg)
![Phoenix](https://img.shields.io/badge/Arize-Phoenix--Observability-orange.svg)

A professional, production-grade **Retrieval-Augmented Generation (RAG)** system designed for academic and corporate research. This assistant goes beyond basic RAG by implementing a 5-level architectural roadmap, including hybrid search, multi-query expansion, and real-time observability.

---

## 🚀 The 5-Level Architecture

This project was built following a professional evolution roadmap to ensure industry-standard performance and reliability:

1.  **Level 1: Precision Retrieval** - Implemented Two-Stage RAG with Cross-Encoder Re-ranking to significantly improve answer accuracy.
2.  **Level 2: Engine Performance** - Migrated to a non-blocking `asyncio` architecture and multi-threaded processing for a responsive UI.
3.  **Level 3: Product Experience** - Added multi-document synthesis and visual PDF page previews for direct source verification.
4.  **Level 4: Observability** - Integrated **Arize Phoenix** for real-time trace monitoring of LLM calls and retrieval loops.
5.  **Level 5: Advanced Search** - Implemented **Hybrid Search** (Vector + BM25) and **Multi-Query Expansion** for state-of-the-art recall.

---

## ✨ Key Features

- **🔍 Hybrid Search Engine**: Combines the semantic power of Vector Search (ChromaDB) with the keyword precision of BM25.
- **🧠 Intelligent Synthesis**: Analyzes your entire document library to find patterns and conflicting information across multiple PDFs.
- **👁️ Visual Citation**: Instantly render the exact PDF page mentioned in an answer for foolproof verification.
- **📡 Developer Tracing**: Live dashboard to monitor every detail of the AI's "thought process" using Arize Phoenix.
- **⚡ Async Architecture**: Built on a solid foundation of Python `asyncio`, making it ready for high-performance usage.

---

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Vector DB**: ChromaDB (Local Persistent Storage)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Search Logic**: Rank-BM25 + Reciprocal Rank Fusion (RRF)
- **PDF Engine**: PyMuPDF (`fitz`)
- **UI Framework**: Streamlit (Premium Custom CSS)
- **Observability**: Arize Phoenix / OpenTelemetry

---

## ⚙️ Installation & Setup

### 1. Clone & Environment
```bash
git clone <your-repo-url>
cd rag-research_assistant
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

---

## 📖 Usage Guide

1.  **Launch the App**:
    ```bash
    streamlit run app.py
    ```
2.  **Index Documents**: Use the sidebar to upload one or multiple PDF research papers.
3.  **Monitor Traces**: Click the "Open Observability Dashboard" button in the sidebar to view live traces.
4.  **Ask Questions**: Type your research query in the main chat. The assistant will perform query expansion, hybrid search, and provide a synthesized answer with visual page citations.

---

## 🧪 Testing

The project includes a comprehensive test suite to ensure reliability:
- **Unit Tests**: `pytest tests/test_pipeline.py`
- **Functional E2E**: `python final_test_suite.py`

---

## 📄 License
Project developed for professional portfolio purposes. See `LICENSE` for details.
