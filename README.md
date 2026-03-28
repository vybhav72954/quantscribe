# QuantScribe

**Automated Thematic Peer Analysis for Indian Banks**

QuantScribe is a retrieval‑augmented (RAG) system and Streamlit app that ingests complex financial disclosures (annual reports, earnings calls, etc.) from Indian banks and converts them into quantifiable risk scores and structured peer comparisons with full citation traceability.

> Live demo: **https://quantscribe-on-verge-of-suicide.streamlit.app/**

---

## What you can do

- **Thematic peer comparison**
  Compare a set of banks on predefined themes such as credit risk, liquidity risk, unsecured lending, capital adequacy, market risk, operational risk, and asset quality trends.

- **Ask free‑text questions over reports**
  Run RAG‑style queries like:
  “Compare HDFC and SBI on GNPA, NNPA, and provisioning with page references.”

- **Bank‑wise risk scoring and summaries**
  View per‑bank risk scores, sentiment scores, qualitative ratings, and executive summaries for a selected theme.

- **Evidence‑backed answers**
  Every answer is grounded in retrieved chunks, with bank name, page number, section header, and excerpt surfaced in the UI.

---

## High‑level architecture

**Offline pipeline**

1. **ETL over PDFs** (`quantscribe/etl/`, `scripts/run_etl.py`)
   - Parse PDFs (narrative + tables), classify pages, detect sections, clean text.
   - Produce typed text chunks with rich metadata.

2. **Chunking** (`quantscribe/chunking/`)
   - Separate narrative and table‑style content.
   - Create semantic chunks with section + page metadata.

3. **Embeddings & FAISS indices** (`quantscribe/embeddings/`, `scripts/kaggle_embed.py`)
   - Embed all chunks using `sentence-transformers/all-MiniLM-L6-v2`.
   - Build one FAISS index per bank/document/fiscal year plus a metadata JSON.

**Online app**

4. **Retrieval & LLM extraction** (`quantscribe/retrieval/`, `quantscribe/llm/`, `app.py`)
   - Load FAISS indices for selected banks.
   - Retrieve top‑k chunks per bank, run Gemini‑based extraction, and assemble peer comparisons.

5. **Visualization** (`app.py`)
   - Streamlit UI for:
     - Theme‑based peer comparison.
     - Free‑text “Ask Reports” RAG QA.
   - Renders rankings, metrics tables, summaries, and evidence previews.

---

## Tech stack

| Component      | Tooling |
| ------------- | ------- |
| UI            | Streamlit, Plotly |
| PDF parsing   | pdfplumber, PyMuPDF, camelot‑py |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store  | FAISS (per‑bank indices under `indices/active`) |
| LLM           | Google Gemini via `langchain-google-genai` |
| Config & env  | `.env` + environment variables, `python-dotenv` |
| Evaluation    | RAGAS, DeepEval, custom numerical checks (optional) |

---

## Getting started (local)

### Prerequisites

- Python **3.10+**
- A Google Gemini API key (`GOOGLE_API_KEY`)
- A Unix‑like environment is recommended for PDF tooling (Camelot, etc.)

### 1. Clone the repo

```bash
git clone https://github.com/ConfusedNeuron/quantscribe.git
cd quantscribe
```

### 2. Create and activate a virtual environment

You can use any tool you like; for example with `uv`:

```bash
uv venv
source .venv/bin/activate  # or equivalent on Windows
```

### 3. Install dependencies

Using `uv` (recommended):

```bash
uv pip install -e ".[dev,eval]"
```

Or with `pip`:

```bash
pip install -e ".[dev,eval]"
```

### 4. Configure environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Key variables (see `.env.example` for the full list):

```env
GOOGLE_API_KEY=your_gemini_api_key_here

PDF_INPUT_DIR=data/pdfs
INDEX_DIR=indices/active
LOG_DIR=logs

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.0
LLM_MAX_RETRIES=3

TOP_K_PER_BANK=5
```

You can also set `GOOGLE_API_KEY` via Streamlit secrets when deploying (the app reads `GOOGLE_API_KEY` from `st.secrets` if available).

---

## Data preparation & index building

The offline pipeline has two core steps:

1. **Run ETL + chunking from PDFs.**
2. **Embed chunks and build FAISS indices.**

After that, the Streamlit app can run purely on the pre‑computed indices plus your Gemini key.

### Step 1 – ETL + chunking from PDFs

Place your bank PDFs under `data/pdfs/`, then run the ETL script for each bank:

```bash
# Example: HDFC Bank annual report FY25
python scripts/run_etl.py \
  --bank HDFC_BANK \
  --pdf "data/pdfs/HDFC Bank Report.pdf" \
  --fy FY25 \
  --doc-type annual_report

# Example: SBI annual report FY25
python scripts/run_etl.py \
  --bank SBI \
  --pdf "data/pdfs/SBI Bank Report.pdf" \
  --fy FY25 \
  --doc-type annual_report

# Optional: process only a subset of pages (for debugging)
python scripts/run_etl.py \
  --bank HDFC_BANK \
  --pdf "data/pdfs/HDFC Bank Report.pdf" \
  --fy FY25 \
  --doc-type annual_report \
  --pages 49-60
```

This will:

- Run the full ETL pipeline (`quantscribe.etl.pipeline.run_etl_pipeline`) over the PDF.
- Produce typed chunks (narrative + tables) with section and page metadata.
- Save them to:

```text
data/chunks/{BANK}_{DOC_TYPE}_{FY}_chunks.json
# e.g. data/chunks/HDFC_BANK_annual_report_FY25_chunks.json
```

The script also prints a quick quality summary:

- Number of chunks and minutes taken.
- Chunk types distribution.
- Sections detected.
- Pages covered.

---

### Step 2 – Embeddings + FAISS indices (Kaggle workflow)

For faster embedding of large chunk sets, use the Kaggle script.

**Workflow:**

1. Upload your `data/chunks/*_chunks.json` files as a Kaggle dataset.
2. Copy `scripts/kaggle_embed.py` into a Kaggle notebook.
3. Enable a GPU accelerator (H100 / T4 / P100).
4. Update `INPUT_DIR` in the script to point to your Kaggle dataset path.
5. Run the notebook.

The script will:

- Load all `*_chunks.json` files.
- Extract `content` for each chunk and embed with `sentence-transformers/all-MiniLM-L6-v2`.
- Build a FAISS `IndexFlatIP` per file.
- Write, for each bank / document / FY:

```text
{BANK}_{DOC_TYPE}_{FY}.faiss
{BANK}_{DOC_TYPE}_{FY}_metadata.json
```

The metadata JSON is critical; it contains:

- The original chunk metadata (page, section, etc.).
- A `content` field with the raw text.
- A `content_type` field (`"narrative"`, `"table_text"`, `"table_structured"`).

> Without `content` in the metadata, the LLM extraction step would receive empty context and hallucinations are more likely.

After the Kaggle run:

- Download all `.faiss` and `_metadata.json` files.
- Place them in your local `indices/active/` folder, for example:

```text
indices/active/
├── HDFC_BANK_annual_report_FY25.faiss
├── HDFC_BANK_annual_report_FY25_metadata.json
├── SBI_annual_report_FY25.faiss
├── SBI_annual_report_FY25_metadata.json
└── ...
```

The Streamlit app and CLI tools will then be able to discover and load these indices.

---

## Running the Streamlit app

Once indices are available under `indices/active/` and you have a valid `GOOGLE_API_KEY`:

```bash
streamlit run app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`) to use the app.

---

## Using the app

### Sidebar controls

- **Mode**
  - `Theme Comparison` – structured extraction for a predefined macro theme across multiple banks.
  - `Ask Reports` – free‑text RAG QA over selected banks.
- **Banks** – multiselect from all banks for which `*.faiss` indices exist in `indices/active/`.
- **Chunks per bank** – number of top‑k chunks retrieved per bank.
- **Fiscal year** – currently `FY25` selector (extend as you add more indices).
- **Theme** (for Theme Comparison) – one of the supported macro themes.
- **Clear cache** – clears Streamlit cached resources and reloads.

### Theme Comparison

- Requires at least **two banks** selected.
- Uses:
  - `EmbeddingPipeline` for query‑time embeddings.
  - `PeerGroupRetriever` to fan‑out retrieval across banks.
  - `run_peer_comparison` for LLM extraction and scoring.

Produces:

- A peer ranking bar chart by risk score (lower is better).
- Summary cards with best‑positioned bank, average risk score, and theme metadata.
- Cross‑cutting insights summarizing the theme across banks.
- Per‑bank detail panels with:
  - Risk score, rating, sentiment score.
  - Executive summary.
  - Extracted metrics table (metric name, value, confidence, page, section).
  - Source excerpts with page, section, and snippet.

### Ask Reports

- Accepts a free‑text question.
- For selected banks:
  - Computes a query embedding via `EmbeddingPipeline`.
  - Uses `PeerGroupRetriever` to fetch top‑k chunks per bank.
  - Builds a consolidated context string with bank / page / section markers.
  - Calls Gemini through `ChatGoogleGenerativeAI` with a retrieval‑grounded prompt.

Shows:

- The answer in a card, with inline bank + page style citations in the text.
- Optional “Retrieved evidence” dataframe listing bank, page, section, type, score, and a short preview.

---

## CLI: full pipeline + evaluation (optional)

For batch runs and evaluation, you can use the CLI scripts under `scripts/`.

### Full pipeline (peer comparison + report JSON)

```bash
# Credit risk peer comparison across HDFC and SBI, using existing indices
python scripts/run_full_pipeline.py \
  --theme credit_risk \
  --banks HDFC_BANK SBI \
  --index-dir indices/active \
  --top-k 5 \
  --skip-eval
```

This will:

- Load FAISS indices for the given banks.
- Run `run_peer_comparison` for the specified theme.
- Print rankings, scores, summaries, and metric details to stdout.
- Save a machine‑readable report to:

```text
data/reports/{theme}_peer_comparison.json
# e.g. data/reports/credit_risk_peer_comparison.json
```

### Evaluation against gold standard

If you maintain gold‑standard cases under `eval/gold_standard/*.json`, you can run the full pipeline **with** evaluation:

```bash
python scripts/run_full_pipeline.py \
  --theme credit_risk \
  --banks HDFC_BANK SBI \
  --index-dir indices/active \
  --top-k 5
```

The script will:

- Load `EvalTestCase` objects from `eval/gold_standard/`.
- Compare extracted metrics against expected values using `evaluate_numerical_accuracy`.
- Print numerical accuracy (PASS/FAIL vs target) and schema‑compliance status.

You can also use `scripts/run_eval.py` for focused evaluation runs.

---

## Project structure

```text
quantscribe/
<<<<<<< HEAD
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
=======
├── app.py                  # Streamlit UI entrypoint
├── quantscribe/            # Core Python package
│   ├── __init__.py
│   ├── config.py           # Settings and path configuration
│   ├── chunking/           # Narrative + table chunking logic
│   ├── embeddings/         # Embedding pipeline and helpers
│   ├── etl/                # PDF parsing, page classification, section detection, cleaning
│   ├── evaluation/         # RAG + extraction evaluation utilities
│   ├── llm/                # Prompts, extraction chain, peer comparison logic
│   ├── logging_config.py   # Structured logging setup
│   ├── retrieval/          # BankIndex + PeerGroupRetriever over FAISS
│   └── schemas/            # Data models for chunks, extractions, evaluation
├── scripts/                # CLI entrypoints (ETL, embeddings, full pipeline, eval)
│   ├── run_etl.py
│   ├── kaggle_embed.py
│   ├── run_full_pipeline.py
│   ├── run_eval.py
│   └── ...
├── indices/                # FAISS index files (active indices under indices/active)
├── eval/
│   └── gold_standard/      # Gold-standard JSON test cases for evaluation
├── docs/                   # Design docs and specs
├── tests/                  # Test suite
├── .streamlit/             # Streamlit configuration
├── .env.example            # Example environment configuration
├── pyproject.toml          # Packaging + dependencies
├── requirements.txt        # Optional extra requirements wrapper
└── LICENSE                 # MIT license
>>>>>>> cdd5486 (Updated README.md)
```

---

## Deployment

The app is currently deployed on **Streamlit Community Cloud** at:

- https://quantscribe-on-verge-of-suicide.streamlit.app/

For your own deployment:

- Set `GOOGLE_API_KEY` as a Streamlit secret.
- Ensure all required FAISS indices and metadata files are available under `indices/active/` in the deployment image.
- Make sure your `pyproject.toml` (or `requirements.txt`) matches the environment on Streamlit Cloud.

---

## Team

Group 8 (IIT Kharagpur):

- Harshita Gaikwad (25BM6JP14)
- Pranav Taneja (25BM6JP37)
- Sneha Yadav (25BM6JP49)
- Srishti Jayaswal (25BM6JP54)
- Vybhav Chaturvedi (25BM6JP60)

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
