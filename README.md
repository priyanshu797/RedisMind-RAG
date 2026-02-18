# Adaptive Hybrid RAG with Redis Semantic Cache

## Overview

This project implements a production-grade Retrieval-Augmented Generation (RAG) pipeline designed for fast, context-aware document question answering. The system combines semantic embedding retrieval, Redis-based vector caching, adaptive query routing, and cross-encoder reranking to improve both performance and response accuracy.

The architecture focuses on reducing latency while maintaining high-quality semantic grounding.

---

## Key Features

* Hybrid Retrieval-Augmented Generation pipeline
* Redis Semantic Cache (L1 memory + L2 Redis HNSW index)
* Adaptive Two-Mode Retrieval Strategy
* Cross-Encoder reranking for relevance optimization
* Context compression for token efficiency
* Persistent vector index storage
* Multi-format document ingestion
* LLM fallback mechanism

---

## Architecture Flow

User Query
→ Query Classification
→ Embedding Generation
→ Redis Semantic Cache Lookup
→ Vector Retrieval
→ Cross-Encoder Reranking
→ Context Compression
→ LLM Response Generation
→ Cache Writeback

---

## Tech Stack

Core Framework:

* Python
* LlamaIndex

Embeddings & Ranking:

* sentence-transformers/all-MiniLM-L6-v2
* cross-encoder/ms-marco-MiniLM-L-6-v2

LLM Integration:

* Groq LLM (Primary)
* Ollama (Fallback)

Caching:

* Redis Stack
* RediSearch HNSW Vector Index

Data Processing:

* PDFPlumber
* Whisper
* Tesseract OCR
* OpenCV

---

## Retrieval Modes

Fast Retrieval Mode:

* Lightweight embedding search
* Optimized for low-latency factual queries

Advanced Reranked Mode:

* Cross-encoder reranking
* Improved contextual accuracy for complex queries

---

## Supported File Types

* PDF
* DOCX
* TXT
* CSV
* JSON
* Images (OCR)
* Audio (Speech-to-Text)
* ZIP archives

---

## Installation

Create virtual environment:

```
python -m venv venv
```

Activate:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Redis Setup

Start Redis Stack locally:

```
redis-stack-server
```

Ensure RediSearch is enabled for vector indexing.

---

## Usage

Run the RAG pipeline:

```
python 1.py
```

Provide file paths when prompted, then ask document-based questions interactively.

---

## Performance Design

* Semantic cache reduces repeated LLM calls
* Adaptive retrieval balances speed and accuracy
* Persistent index prevents unnecessary recomputation

---

## Security Notes

* Store API keys in `.env`
* Do not expose Redis publicly
* Avoid committing sensitive data

---

## License

This project is intended for educational and development use.
