# QuantScribe

**Automated Thematic Peer Analysis for Indian BFSI Sector**

QuantScribe is a RAG-based pipeline that ingests complex financial disclosures (Annual Reports, Earnings Calls) from Indian banks and converts them into quantifiable risk scores and structured alpha signals with full citation traceability.

## Architecture

```
PDF Documents → ETL & Page Classification → Semantic Chunking → Embedding → FAISS Index
                                                                                    ↓
User Query → Query Embedding → Fan-out Retrieval (per bank) → LLM Extraction → Peer Report
```

## Core Design Principles

1. **Deterministic extraction** — identical inputs produce identical outputs
2. **Zero cross-entity contamination** — Bank A's metrics never bleed into Bank B's analysis
3. **Full citation traceability** — every metric maps to a specific source chunk
4. **Graceful degradation** — parsing failures are logged, never silently swallowed

## Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | pdfplumber (tables) + PyMuPDF (narrative) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS IndexFlatIP |
| LLM | Google Gemini 2.5 Flash via API |
| Orchestration | LangChain |
| Schema Enforcement | Pydantic v2 |
| Evaluation | RAGAS + DeepEval |

## Quick Start

All the [Data](https://drive.google.com/drive/u/5/folders/1KVUonddJzhCiYDgTqJ09Sk_ngi9Wnupz) is available in the google drive link.
Please make sure you download the data and place it in the `data/pdfs` folder. (Create a new folder if required)

```bash
# Clone
git clone https://github.com/vybhav72954/quantscribe.git
cd quantscribe
# Please Create venv beforehand, i recommned using uv, youcan use anything else as well.

# Install
uv pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your Gemini API key

# Run tests
pytest tests/ -v
```

## Project Structure

```
quantscribe/
├── quantscribe/
│   ├── config.py           # Settings, theme taxonomy, known sections
│   ├── schemas/             # Pydantic models (the system contract)
│   ├── etl/                 # PDF parsing, page classification, text cleaning
│   ├── chunking/            # Narrative + table chunking with metadata
│   ├── embeddings/          # Embedding pipeline with L2 normalization
│   ├── retrieval/           # BankIndex, PeerGroupRetriever
│   ├── llm/                 # Prompts, extraction chain, peer comparison
│   └── evaluation/          # Numerical accuracy, RAGAS, DeepEval
├── scripts/
├── eval/gold_standard/      # Ground-truth test cases
├── indices/                 # FAISS indices (git-ignored)
├── tests/                   # Test suite
├── data/pdfs/               # Data folder (git-ignored)
├── docs/                    # Specification documents
└── logs/                    # Logs (git-ignored)
```

## Development Phases

| Phase | Description |
|---|---|
| Phase 0 | Repo scaffold, schemas, config |
| Phase 1 | ETL & page classification |
| Phase 2 | Chunking & metadata |
| Phase 3 | Embedding & FAISS indexing |
| Phase 4 | LLM extraction & peer comparison |
| Phase 5 | Evaluation suite |

## Team

Group 8:
- Harshita Gaikwad (25BM6JP14)
- Pranav Taneja (25BM6JP37)
- Sneha Yadav (25BM6JP49)
- Srishti Jayaswal (25BM6JP54)
- Vybhav Chaturvedi (25BM6JP60)
