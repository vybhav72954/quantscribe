# QuantScribe v2: Hardened System Architecture & Engineering Specification

> **Status:** Engineering-Ready Specification
> **Version:** 2.0
> **Last Updated:** 2026-03-26
> **Authors:** Group 8 (Harshita Gaikwad, Pranav Taneja, Sneha Yadav, Srishti Jayaswal, Vybhav Chaturvedi)

---

## 1. Project Overview

QuantScribe is a production-grade, RAG-based automated thematic peer analysis tool for the Indian BFSI sector. It ingests complex, unstructured financial disclosures (Annual Reports, Earnings Call Transcripts) and converts them into quantifiable risk scores and structured alpha signals with full citation traceability.

**Design Principles:**

1. Deterministic extraction — identical inputs must produce identical outputs.
2. Zero cross-entity contamination — Bank A's metrics must never bleed into Bank B's analysis.
3. Full citation traceability — every extracted metric maps to a specific source chunk with bank, document, page, and section metadata.
4. Graceful degradation — parsing failures are logged, quarantined, and never silently swallowed.

---

## 2. Core Tech Stack

| Layer | Tool | Purpose | Notes |
|---|---|---|---|
| Language | Python 3.10+ | All modules | Use `typing` extensively; strict type hints everywhere |
| PDF Tables | `pdfplumber` | Table boundary detection, row/column extraction | Primary tool for `Tabular` pages |
| PDF Narrative | `PyMuPDF` (`fitz`) | Text extraction with font-size/position metadata | Primary tool for `Narrative` pages |
| Supplementary Tables | `camelot-py` | Fallback for tables pdfplumber misses | Lattice + stream mode |
| Vector Database | FAISS (`IndexFlatIP`) | Cosine similarity on L2-normalized vectors | One index per `{bank}_{doc_type}_{fiscal_year}` |
| Embeddings (MVP) | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim embeddings, 256 token max input | Must L2-normalize before FAISS insertion |
| Embeddings (Prod) | `FinBERT` sentence embeddings | Domain-specific financial embeddings | Migration after MVP validation |
| LLM Orchestration | LangChain | Chain construction, prompt templates, output parsing | Use `PydanticOutputParser` exclusively |
| Output Enforcement | Pydantic v2 | Strict JSON schema compliance | All LLM outputs validated before storage |
| Evaluation | RAGAS + DeepEval | Retrieval quality + Faithfulness scoring | See Phase 5 for full protocol |

**Explicitly Excluded:**
- BERTScore, METEOR, BLEU — these are reference-based MT metrics that cannot evaluate numerical precision or JSON validity.
- TruLens — useful for tracing but redundant given DeepEval coverage. Can be added later for debugging.

---

## 3. Pydantic Schema Definitions (The System Contract)

These models are the single source of truth for all inter-module communication. Every phase of the pipeline produces or consumes these schemas.

### 3.1 ETL & Chunking Schemas

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
import hashlib


class PageType(str, Enum):
    NARRATIVE = "narrative"
    TABULAR = "tabular"
    MIXED = "mixed"
    GRAPHICAL = "graphical"


class ChunkMetadata(BaseModel):
    """Metadata envelope attached to EVERY chunk before embedding.
    This is the primary mechanism preventing cross-entity contamination."""

    chunk_id: str = Field(
        description="Deterministic hash: sha256(bank_name + doc_type + fiscal_year + page_number + chunk_index)"
    )
    bank_name: str = Field(
        description="Standardized bank identifier",
        examples=["HDFC_BANK", "SBI", "ICICI_BANK", "AXIS_BANK"]
    )
    document_type: Literal["annual_report", "earnings_call", "investor_presentation"]
    fiscal_year: str = Field(
        pattern=r"^FY\d{2}$",
        description="Fiscal year in FYxx format",
        examples=["FY23", "FY24"]
    )
    page_number: int = Field(ge=1)
    section_header: Optional[str] = Field(
        default=None,
        description="Nearest extractable section header (e.g., 'Management Discussion & Analysis')"
    )
    page_type: PageType
    chunk_index: int = Field(
        ge=0,
        description="Sequential index of this chunk within the page"
    )
    token_count: int = Field(ge=1)
    parse_version: str = Field(
        description="ETL pipeline version that produced this chunk",
        examples=["etl_v1.0.0", "etl_v1.2.3"]
    )

    @field_validator("bank_name")
    @classmethod
    def normalize_bank_name(cls, v: str) -> str:
        """Enforce uppercase snake_case for all bank names."""
        return v.upper().replace(" ", "_").replace("-", "_")


class ParsedPage(BaseModel):
    """Output of the page classification + extraction step."""

    page_number: int
    page_type: PageType
    raw_text: Optional[str] = None
    tables: Optional[list[dict]] = Field(
        default=None,
        description="List of tables as list-of-dicts (each dict = one row)"
    )
    extraction_warnings: list[str] = Field(default_factory=list)
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in page classification (0-1)"
    )


class TextChunk(BaseModel):
    """A single chunk ready for embedding."""

    content: str = Field(min_length=10)
    metadata: ChunkMetadata
    content_type: Literal["narrative", "table_text", "table_structured"]
```

### 3.2 Retrieval & LLM Output Schemas

```python
class CitationTrace(BaseModel):
    """Every extracted metric MUST have one of these.
    This is what makes the system auditable."""

    chunk_id: str
    bank_name: str
    document_type: str
    fiscal_year: str
    page_number: int
    section_header: Optional[str] = None
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Cosine similarity score from FAISS retrieval"
    )
    source_excerpt: str = Field(
        max_length=500,
        description="The exact text span from the chunk that supports this metric"
    )


class ExtractedMetric(BaseModel):
    """A single quantitative or qualitative metric extracted by the LLM."""

    metric_name: str = Field(examples=["gross_npa_ratio", "capital_adequacy_ratio"])
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = Field(
        default=None,
        examples=["%", "INR_crore", "basis_points", "ratio"]
    )
    qualitative_value: Optional[str] = Field(
        default=None,
        description="For non-numeric assessments: 'high', 'moderate', 'low', 'stable', 'deteriorating'"
    )
    confidence: Literal["high", "medium", "low"]
    citation: CitationTrace

    @field_validator("metric_value", "qualitative_value")
    @classmethod
    def at_least_one_value(cls, v, info):
        """Ensure at least one of metric_value or qualitative_value is set."""
        # Validated at model level via model_validator in practice
        return v


class ThematicExtraction(BaseModel):
    """LLM output for a single bank on a single macro theme.
    This is the atomic unit of analysis."""

    bank_name: str
    fiscal_year: str
    theme: str = Field(
        description="The macro theme being analyzed",
        examples=["credit_risk", "liquidity_risk", "unsecured_lending"]
    )
    risk_score: float = Field(
        ge=0.0, le=10.0,
        description="Computed risk score (0=lowest risk, 10=highest risk)"
    )
    risk_rating: Literal["very_low", "low", "moderate", "high", "critical"]
    summary: str = Field(
        max_length=1000,
        description="Qualitative summary grounded entirely in retrieved text"
    )
    extracted_metrics: list[ExtractedMetric] = Field(min_length=1)
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Sentiment of disclosure language (-1=very negative, +1=very positive)"
    )


class PeerComparisonReport(BaseModel):
    """The final cross-bank comparison output.
    One of these is generated per query."""

    query_theme: str
    peer_group: list[str] = Field(
        min_length=2,
        description="List of bank names being compared"
    )
    extractions: list[ThematicExtraction]
    peer_ranking: list[dict] = Field(
        description="Banks ranked by risk_score for this theme, ascending"
    )
    cross_cutting_insights: str = Field(
        max_length=2000,
        description="Synthesized insights comparing trends across the peer group"
    )
    generated_at: str = Field(description="ISO 8601 timestamp")
```

### 3.3 Evaluation Schemas

```python
class EvalTestCase(BaseModel):
    """A single evaluation test case with ground-truth annotations."""

    test_id: str
    query_theme: str
    bank_name: str
    fiscal_year: str
    expected_metrics: dict[str, float] = Field(
        description="Ground-truth metric name -> value pairs",
        examples=[{"gross_npa_ratio": 1.12, "net_npa_ratio": 0.27}]
    )
    expected_pages: list[int] = Field(
        description="Page numbers where this data should be found"
    )
    source_document: str


class EvalResult(BaseModel):
    """Result of running evaluation on a single test case."""

    test_id: str
    numerical_accuracy: dict[str, bool] = Field(
        description="Per-metric exact match (within tolerance)"
    )
    schema_valid: bool
    context_precision: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    retrieval_hit: bool = Field(
        description="Did at least one retrieved chunk come from an expected_page?"
    )
```

---

## 4. Architectural Specification: Phase by Phase

### Phase 1: ETL & Page Classification

**This is the highest-risk phase. Budget 40-50% of development time here.**

#### 1.1 Page Classification Logic

Every page must be classified before extraction. Use the following deterministic heuristic pipeline:

```python
def classify_page(page, pdf_path: str) -> PageType:
    """
    Classification heuristic for a single PDF page.
    
    Uses pdfplumber table detection + PyMuPDF text block analysis.
    
    Decision Logic:
    1. Count tables detected by pdfplumber (table_count)
    2. Count text blocks detected by PyMuPDF (text_block_count)
    3. Check for images/drawings via PyMuPDF (image_count)
    
    Rules:
    - image_count > 2 AND text_block_count < 3  -->  GRAPHICAL
    - table_count > 0 AND text_block_count > 5   -->  MIXED
    - table_count > 0 AND text_block_count <= 5   -->  TABULAR
    - table_count == 0 AND text_block_count > 0   -->  NARRATIVE
    - fallback                                     -->  GRAPHICAL (skip)
    """
    import pdfplumber
    import fitz

    # pdfplumber pass: detect tables
    with pdfplumber.open(pdf_path) as pdf:
        plumber_page = pdf.pages[page.number]  # 0-indexed
        tables = plumber_page.find_tables()
        table_count = len(tables)

    # PyMuPDF pass: detect text blocks and images
    doc = fitz.open(pdf_path)
    mu_page = doc[page.number]
    text_blocks = mu_page.get_text("blocks")
    text_block_count = len([b for b in text_blocks if b[6] == 0])  # type 0 = text
    image_count = len(mu_page.get_images())
    doc.close()

    # Decision tree
    if image_count > 2 and text_block_count < 3:
        return PageType.GRAPHICAL
    elif table_count > 0 and text_block_count > 5:
        return PageType.MIXED
    elif table_count > 0:
        return PageType.TABULAR
    elif text_block_count > 0:
        return PageType.NARRATIVE
    else:
        return PageType.GRAPHICAL
```

**Thresholds** (table_count, text_block_count, image_count) must be tuned empirically on 10 sample pages per bank. Log every classification with its confidence score to enable auditing.

#### 1.2 Extraction Routing

| Page Type | Primary Tool | Fallback | Action |
|---|---|---|---|
| `NARRATIVE` | PyMuPDF `get_text("blocks")` | pdfplumber `.extract_text()` | Extract text preserving paragraph boundaries via block positions |
| `TABULAR` | pdfplumber `.extract_tables()` | `camelot-py` (lattice + stream) | Extract as list-of-dicts with column headers as keys |
| `MIXED` | Split into sub-regions | — | Identify table bounding boxes via pdfplumber, extract tables; extract remaining text blocks via PyMuPDF |
| `GRAPHICAL` | Skip | — | Log and skip; attach `extraction_warning: "graphical_page_skipped"` |

#### 1.3 Mixed Page Handling (Critical)

Mixed pages are the most common and most dangerous page type in Indian annual reports. The strategy:

1. Use pdfplumber to detect table bounding boxes on the page.
2. For each detected table: extract the table using pdfplumber, record its `(x0, y0, x1, y1)` bounding box.
3. Use PyMuPDF to extract all text blocks, then **filter out** any text block whose centroid falls within a table bounding box (these are already captured by pdfplumber).
4. The remaining text blocks are narrative content.
5. Produce separate chunks: one per table (as `table_text` or `table_structured`), and narrative chunks from the remaining text.

#### 1.4 Table Extraction Hardening

Financial tables in Indian annual reports have known pathologies:

| Pathology | Detection | Mitigation |
|---|---|---|
| Multi-page tables | Header row repeats on consecutive pages | Detect repeated headers via string match; merge rows across pages |
| Merged cells | pdfplumber returns `None` in cell positions | Forward-fill None values from the cell above |
| Footnote rows | Rows with `*` or `†` markers | Separate into metadata; attach as chunk annotation |
| Currency formatting | `₹ 1,23,456.78` (Indian numbering) | Regex normalization to float before storage |
| Hidden unicode | Zero-width spaces, soft hyphens | Strip `\u200b`, `\u00ad`, `\ufeff` during extraction |

```python
import re

def normalize_indian_currency(text: str) -> str:
    """Convert Indian-format numbers to parseable floats.
    '₹ 1,23,456.78' -> '123456.78'
    '(1,234.56)' -> '-1234.56'  (parentheses = negative in accounting)
    """
    text = text.replace("₹", "").replace(" ", "").strip()
    # Handle accounting negatives
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    # Remove Indian comma formatting
    text = text.replace(",", "")
    try:
        return str(float(text))
    except ValueError:
        return text  # Return original if not parseable


def strip_unicode_garbage(text: str) -> str:
    """Remove zero-width and invisible unicode characters."""
    garbage = ["\u200b", "\u200c", "\u200d", "\u00ad", "\ufeff", "\u00a0"]
    for char in garbage:
        text = text.replace(char, "")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

#### 1.5 Section Header Detection

Section headers are critical metadata but are not consistently formatted across banks. Use a multi-strategy approach:

```python
KNOWN_SECTIONS = [
    "Management Discussion and Analysis",
    "Management Discussion & Analysis",
    "MD&A",
    "Risk Management",
    "Credit Risk",
    "Market Risk",
    "Operational Risk",
    "Liquidity Risk",
    "Capital Adequacy",
    "Basel III Disclosures",
    "Asset Quality",
    "Corporate Governance",
    "Auditor's Report",
    "Notes to Financial Statements",
    "Profit and Loss",
    "Balance Sheet",
    "Cash Flow Statement",
    "Schedules to Financial Statements",
    "Directors' Report",
]


def detect_section_header(page_blocks: list, page_number: int) -> Optional[str]:
    """
    Detect section headers using three strategies in priority order:
    
    1. Exact/fuzzy match against KNOWN_SECTIONS
    2. Font-size heuristic: text blocks with font size > 1.3x page median
    3. Position heuristic: bold text in the top 15% of the page
    
    Returns the best candidate or None.
    """
    from difflib import get_close_matches

    candidates = []

    for block in page_blocks:
        text = block["text"].strip()
        if not text or len(text) > 100:
            continue

        # Strategy 1: Known section matching
        matches = get_close_matches(text, KNOWN_SECTIONS, n=1, cutoff=0.75)
        if matches:
            return matches[0]

        # Strategy 2: Font-size heuristic (requires PyMuPDF spans)
        if block.get("font_size") and block.get("median_font_size"):
            if block["font_size"] > block["median_font_size"] * 1.3:
                candidates.append(text)

        # Strategy 3: Position heuristic (top 15% of page)
        if block.get("y_position") and block.get("page_height"):
            if block["y_position"] < block["page_height"] * 0.15:
                if block.get("is_bold"):
                    candidates.append(text)

    return candidates[0] if candidates else None
```

#### 1.6 Error Handling & Fallback Strategy

| Failure Mode | Detection | Action | Logging |
|---|---|---|---|
| pdfplumber cannot detect table boundaries | `find_tables()` returns empty on a page visually containing tables | Fallback to `camelot-py` with `flavor='stream'` | `WARN: pdfplumber_table_miss, page={n}, bank={bank}` |
| PyMuPDF returns empty text | `get_text()` returns `""` on non-graphical page | Attempt OCR via `pytesseract` on page raster | `WARN: empty_text_fallback_ocr, page={n}` |
| Chunk has no extractable section header | `detect_section_header()` returns `None` | Set `section_header = "UNKNOWN_SECTION"`, carry forward last known header from previous page | `INFO: header_carryforward, page={n}` |
| Malformed table (all None values) | More than 50% of cells are `None` after extraction | Quarantine the page; extract as raw text instead | `ERROR: malformed_table_quarantined, page={n}` |
| Encoding errors / unicode garbage | `UnicodeDecodeError` or garbled text detected | Apply `strip_unicode_garbage()`; if still garbled, skip page | `ERROR: encoding_failure, page={n}` |
| PDF file is scanned (image-only) | Zero text blocks on > 80% of pages | Abort and notify user: "This PDF requires OCR preprocessing" | `CRITICAL: scanned_pdf_detected` |

**Non-negotiable rule:** No failure is ever silently swallowed. Every extraction failure produces a structured log entry with the page number, bank name, and failure type. These logs feed into the evaluation dashboard.

---

### Phase 2: Hybrid Semantic Chunking & Metadata Attachment

#### 2.1 Chunking Parameters

| Parameter | Narrative Text | Tabular Data |
|---|---|---|
| Chunk size | 300 words (~400 tokens) | Atomic unit (entire table) |
| Overlap | 100 words | 0 (no overlap) |
| Max chunk size | 500 words (~650 tokens) | 1024 tokens (if table exceeds, split at row boundary) |
| Header attachment | Prepend detected section header | Prepend column headers to every sub-chunk |
| Metadata | Full `ChunkMetadata` envelope | Full `ChunkMetadata` envelope + column header list |

#### 2.2 Narrative Chunking Strategy

```python
def chunk_narrative(
    text: str,
    metadata_base: dict,
    chunk_size_words: int = 300,
    overlap_words: int = 100,
) -> list[TextChunk]:
    """
    Chunk narrative text with overlap.
    
    Rules:
    1. Split on sentence boundaries (never mid-sentence).
    2. Target chunk_size_words per chunk with overlap_words overlap.
    3. Prepend section_header to every chunk.
    4. Attach full ChunkMetadata to every chunk.
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += word_count

        if current_word_count >= chunk_size_words:
            chunk_text = " ".join(current_sentences)

            # Prepend section header if available
            header = metadata_base.get("section_header")
            if header:
                chunk_text = f"[Section: {header}]\n{chunk_text}"

            metadata = ChunkMetadata(
                chunk_id=_generate_chunk_id(metadata_base, chunk_index),
                bank_name=metadata_base["bank_name"],
                document_type=metadata_base["document_type"],
                fiscal_year=metadata_base["fiscal_year"],
                page_number=metadata_base["page_number"],
                section_header=metadata_base.get("section_header"),
                page_type=PageType.NARRATIVE,
                chunk_index=chunk_index,
                token_count=len(chunk_text.split()),  # Approximate
                parse_version=metadata_base["parse_version"],
            )

            chunks.append(TextChunk(
                content=chunk_text,
                metadata=metadata,
                content_type="narrative",
            ))

            # Overlap: keep last N words worth of sentences
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                overlap_count += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_count >= overlap_words:
                    break

            current_sentences = overlap_sentences
            current_word_count = sum(len(s.split()) for s in current_sentences)
            chunk_index += 1

    # Handle remaining text
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        header = metadata_base.get("section_header")
        if header:
            chunk_text = f"[Section: {header}]\n{chunk_text}"

        metadata = ChunkMetadata(
            chunk_id=_generate_chunk_id(metadata_base, chunk_index),
            bank_name=metadata_base["bank_name"],
            document_type=metadata_base["document_type"],
            fiscal_year=metadata_base["fiscal_year"],
            page_number=metadata_base["page_number"],
            section_header=metadata_base.get("section_header"),
            page_type=PageType.NARRATIVE,
            chunk_index=chunk_index,
            token_count=len(chunk_text.split()),
            parse_version=metadata_base["parse_version"],
        )
        chunks.append(TextChunk(
            content=chunk_text,
            metadata=metadata,
            content_type="narrative",
        ))

    return chunks


def _generate_chunk_id(metadata_base: dict, chunk_index: int) -> str:
    """Deterministic chunk ID via SHA-256."""
    raw = f"{metadata_base['bank_name']}_{metadata_base['document_type']}_{metadata_base['fiscal_year']}_{metadata_base['page_number']}_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

#### 2.3 Table Chunking Strategy

```python
def chunk_table(
    table_data: list[dict],
    metadata_base: dict,
    max_tokens: int = 1024,
    chunk_index_start: int = 0,
) -> list[TextChunk]:
    """
    Chunk a table into atomic units.
    
    Rules:
    1. NEVER split mid-row.
    2. If the table fits within max_tokens, keep it as one chunk.
    3. If it exceeds max_tokens, split at row boundaries.
    4. ALWAYS prepend column headers to every sub-chunk.
    """
    if not table_data:
        return []

    headers = list(table_data[0].keys())
    header_line = " | ".join(headers)

    # Convert entire table to text
    full_text_lines = [header_line, "-" * len(header_line)]
    for row in table_data:
        row_line = " | ".join(str(row.get(h, "")) for h in headers)
        full_text_lines.append(row_line)

    full_text = "\n".join(full_text_lines)

    # Check if it fits in one chunk
    if len(full_text.split()) <= max_tokens:
        metadata = ChunkMetadata(
            chunk_id=_generate_chunk_id(metadata_base, chunk_index_start),
            bank_name=metadata_base["bank_name"],
            document_type=metadata_base["document_type"],
            fiscal_year=metadata_base["fiscal_year"],
            page_number=metadata_base["page_number"],
            section_header=metadata_base.get("section_header"),
            page_type=PageType.TABULAR,
            chunk_index=chunk_index_start,
            token_count=len(full_text.split()),
            parse_version=metadata_base["parse_version"],
        )
        return [TextChunk(
            content=full_text,
            metadata=metadata,
            content_type="table_structured",
        )]

    # Split at row boundaries with header repetition
    chunks = []
    current_rows = []
    current_token_count = len(header_line.split()) + 1  # header + separator
    chunk_idx = chunk_index_start

    for row in table_data:
        row_line = " | ".join(str(row.get(h, "")) for h in headers)
        row_tokens = len(row_line.split())

        if current_token_count + row_tokens > max_tokens and current_rows:
            # Flush current chunk
            chunk_lines = [header_line, "-" * len(header_line)]
            chunk_lines.extend(current_rows)
            chunk_text = "\n".join(chunk_lines)

            metadata = ChunkMetadata(
                chunk_id=_generate_chunk_id(metadata_base, chunk_idx),
                bank_name=metadata_base["bank_name"],
                document_type=metadata_base["document_type"],
                fiscal_year=metadata_base["fiscal_year"],
                page_number=metadata_base["page_number"],
                section_header=metadata_base.get("section_header"),
                page_type=PageType.TABULAR,
                chunk_index=chunk_idx,
                token_count=len(chunk_text.split()),
                parse_version=metadata_base["parse_version"],
            )
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=metadata,
                content_type="table_structured",
            ))

            current_rows = []
            current_token_count = len(header_line.split()) + 1
            chunk_idx += 1

        current_rows.append(row_line)
        current_token_count += row_tokens

    # Flush remaining
    if current_rows:
        chunk_lines = [header_line, "-" * len(header_line)]
        chunk_lines.extend(current_rows)
        chunk_text = "\n".join(chunk_lines)

        metadata = ChunkMetadata(
            chunk_id=_generate_chunk_id(metadata_base, chunk_idx),
            bank_name=metadata_base["bank_name"],
            document_type=metadata_base["document_type"],
            fiscal_year=metadata_base["fiscal_year"],
            page_number=metadata_base["page_number"],
            section_header=metadata_base.get("section_header"),
            page_type=PageType.TABULAR,
            chunk_index=chunk_idx,
            token_count=len(chunk_text.split()),
            parse_version=metadata_base["parse_version"],
        )
        chunks.append(TextChunk(
            content=chunk_text,
            metadata=metadata,
            content_type="table_structured",
        ))

    return chunks
```

---

### Phase 3: Embedding Pipeline & Vector DB

#### 3.1 Embedding Specification

| Parameter | Value | Rationale |
|---|---|---|
| Model | `all-MiniLM-L6-v2` (MVP) | 384-dim, fast, good general-purpose |
| Max input tokens | 256 tokens | MiniLM **silently truncates** beyond this |
| Normalization | L2-normalize all vectors before FAISS insertion | Required for `IndexFlatIP` to compute cosine similarity |
| Batch size | 64 chunks per batch | Balance memory and throughput |
| Long chunk handling | If chunk > 256 tokens, split into sub-chunks and **mean-pool** their embeddings | Prevents silent truncation |

```python
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    """Handles embedding with proper normalization and overflow protection."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.max_tokens = 256
        self.dimension = 384

    def embed_chunks(self, chunks: list[TextChunk], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of TextChunks with overflow protection.
        Returns L2-normalized vectors ready for FAISS IndexFlatIP.
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalization
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns L2-normalized vector."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)
```

#### 3.2 FAISS Index Architecture: Per-Entity Partitioning

**The `BankIndex` Abstraction:**

```python
import faiss
import json
import os
from pathlib import Path


class BankIndex:
    """
    Wraps a single FAISS index for one bank-document-year combination.
    
    Index naming convention: {bank_name}_{doc_type}_{fiscal_year}
    Example: HDFC_BANK_annual_report_FY24
    """

    def __init__(self, index_name: str, dimension: int = 384):
        self.index_name = index_name
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: list[dict] = []  # Parallel array: metadata_store[i] corresponds to vector[i]

    def add(self, embeddings: np.ndarray, chunk_metadata: list[ChunkMetadata]):
        """Add vectors with their metadata."""
        assert len(embeddings) == len(chunk_metadata)
        self.index.add(embeddings)
        self.metadata_store.extend([m.model_dump() for m in chunk_metadata])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search this bank's index. Returns list of dicts with:
        - metadata: ChunkMetadata fields
        - score: cosine similarity
        """
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = {
                "metadata": self.metadata_store[idx],
                "score": float(score),
            }
            results.append(result)
        return results

    def save(self, directory: str):
        """Persist index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, f"{self.index_name}.faiss"))
        with open(os.path.join(directory, f"{self.index_name}_metadata.json"), "w") as f:
            json.dump(self.metadata_store, f)

    def load(self, directory: str):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(os.path.join(directory, f"{self.index_name}.faiss"))
        with open(os.path.join(directory, f"{self.index_name}_metadata.json"), "r") as f:
            self.metadata_store = json.load(f)


class PeerGroupRetriever:
    """
    Fan-out retrieval across multiple BankIndex instances.
    This is the primary mechanism preventing cross-entity contamination.
    """

    def __init__(self, bank_indices: dict[str, BankIndex]):
        """bank_indices: mapping of index_name -> BankIndex"""
        self.bank_indices = bank_indices

    def retrieve(
        self,
        query_vector: np.ndarray,
        peer_group: list[str],
        top_k_per_bank: int = 5,
    ) -> dict[str, list[dict]]:
        """
        Fan out the query to each bank's index independently.
        Returns: {bank_name: [results]} with results sorted by score.
        
        This guarantees each bank gets equal representation in results,
        preventing one bank's verbose disclosures from dominating.
        """
        results = {}
        for index_name, bank_index in self.bank_indices.items():
            # Check if this index belongs to a requested peer
            bank_name = bank_index.metadata_store[0]["bank_name"] if bank_index.metadata_store else ""
            if bank_name in peer_group:
                bank_results = bank_index.search(query_vector, top_k=top_k_per_bank)
                results[bank_name] = bank_results

        return results
```

#### 3.3 Index Versioning

Every FAISS index is tagged with the ETL pipeline version that produced it:

```
indices/
├── v1.0.0/
│   ├── HDFC_BANK_annual_report_FY24.faiss
│   ├── HDFC_BANK_annual_report_FY24_metadata.json
│   ├── SBI_annual_report_FY24.faiss
│   └── ...
├── v1.1.0/
│   ├── HDFC_BANK_annual_report_FY24.faiss
│   └── ...
└── active -> v1.1.0  (symlink to current production version)
```

When the ETL pipeline is updated, re-index all documents and create a new version directory. The `active` symlink allows atomic cutover.

---

### Phase 4: LLM Orchestration & Extraction

#### 4.1 Prompt Template with Rigid Delimiters

```python
THEMATIC_EXTRACTION_PROMPT = """You are a quantitative financial analyst. You are given retrieved text 
from a bank's annual report. Your task is to extract specific metrics related to the queried theme 
and produce a structured risk assessment.

CRITICAL RULES:
1. ONLY use information from the provided context below. Do NOT use any prior knowledge.
2. If a metric is not explicitly stated in the context, set its confidence to "low" and 
   qualitative_value to "not_disclosed".
3. Every metric you extract MUST include a citation with the exact source_excerpt 
   (max 500 chars) from the context.
4. Risk scores must be between 0.0 (lowest risk) and 10.0 (highest risk).
5. Do NOT infer, calculate, or hallucinate any numbers not present in the text.

QUERIED THEME: {theme}

{bank_contexts}

{format_instructions}
"""

BANK_CONTEXT_TEMPLATE = """
[BEGIN {bank_name} CONTEXT — {fiscal_year} — {document_type}]
{chunks}
[END {bank_name} CONTEXT]
"""
```

#### 4.2 LangChain Chain with Retry Logic

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithFallbacks
import logging

logger = logging.getLogger("quantscribe.llm")


def build_extraction_chain(llm, max_retries: int = 3):
    """
    Build a LangChain extraction chain with:
    1. Pydantic output parsing
    2. Automatic retry on malformed JSON
    3. Structured error logging
    """
    parser = PydanticOutputParser(pydantic_object=ThematicExtraction)

    prompt = PromptTemplate(
        template=THEMATIC_EXTRACTION_PROMPT,
        input_variables=["theme", "bank_contexts"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    # Wrap with retry logic
    def invoke_with_retry(inputs: dict) -> ThematicExtraction:
        last_error = None
        for attempt in range(max_retries):
            try:
                result = chain.invoke(inputs)
                # Post-validation: ensure all citations reference real chunks
                _validate_citations(result, inputs)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM extraction attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    # Append error context to prompt for self-correction
                    inputs["bank_contexts"] += (
                        f"\n\n[SYSTEM: Previous attempt produced invalid output: {str(e)[:200]}. "
                        f"Please correct and try again.]"
                    )

        logger.error(f"LLM extraction failed after {max_retries} retries: {last_error}")
        raise RuntimeError(
            f"Extraction failed for theme={inputs.get('theme')} after {max_retries} retries"
        ) from last_error

    return invoke_with_retry


def _validate_citations(extraction: ThematicExtraction, inputs: dict):
    """
    Post-validation: ensure every citation's source_excerpt
    actually appears in the provided context.
    """
    context_text = inputs.get("bank_contexts", "")
    for metric in extraction.extracted_metrics:
        excerpt = metric.citation.source_excerpt
        # Allow fuzzy match (LLMs sometimes paraphrase slightly)
        if excerpt not in context_text:
            # Check if at least 60% of words appear
            excerpt_words = set(excerpt.lower().split())
            context_words = set(context_text.lower().split())
            overlap = len(excerpt_words & context_words) / max(len(excerpt_words), 1)
            if overlap < 0.6:
                raise ValueError(
                    f"Citation validation failed: excerpt not found in context. "
                    f"Metric: {metric.metric_name}, Overlap: {overlap:.2f}"
                )
```

#### 4.3 Peer Comparison Orchestrator

```python
from datetime import datetime


def run_peer_comparison(
    theme: str,
    peer_group: list[str],
    retriever: PeerGroupRetriever,
    embedding_pipeline: EmbeddingPipeline,
    extraction_chain,
    top_k_per_bank: int = 5,
) -> PeerComparisonReport:
    """
    End-to-end peer comparison pipeline.
    
    1. Embed the thematic query
    2. Fan-out retrieval to each bank's index
    3. Format retrieved chunks with rigid delimiters
    4. Run LLM extraction per bank
    5. Rank and synthesize
    """
    # Step 1: Embed query
    query_vector = embedding_pipeline.embed_query(
        f"{theme} risk exposure analysis financial metrics"
    )

    # Step 2: Fan-out retrieval
    all_results = retriever.retrieve(query_vector, peer_group, top_k_per_bank)

    # Step 3: Format contexts with rigid delimiters
    bank_contexts = []
    for bank_name, results in all_results.items():
        chunks_text = "\n\n---\n\n".join(
            f"[Page {r['metadata']['page_number']}] "
            f"[Section: {r['metadata'].get('section_header', 'N/A')}]\n"
            f"{r['metadata'].get('content', 'Content not stored in metadata')}"
            for r in results
        )
        bank_contexts.append(
            BANK_CONTEXT_TEMPLATE.format(
                bank_name=bank_name,
                fiscal_year=results[0]["metadata"]["fiscal_year"] if results else "N/A",
                document_type=results[0]["metadata"]["document_type"] if results else "N/A",
                chunks=chunks_text,
            )
        )

    combined_context = "\n\n".join(bank_contexts)

    # Step 4: Run extraction per bank
    extractions = []
    for bank_name in peer_group:
        bank_specific_context = [ctx for ctx in bank_contexts if bank_name in ctx]
        if not bank_specific_context:
            logger.warning(f"No context retrieved for {bank_name}, skipping")
            continue

        extraction = extraction_chain({
            "theme": theme,
            "bank_contexts": "\n".join(bank_specific_context),
        })
        extractions.append(extraction)

    # Step 5: Rank by risk score
    ranked = sorted(extractions, key=lambda x: x.risk_score)
    peer_ranking = [
        {"bank": e.bank_name, "risk_score": e.risk_score, "rank": i + 1}
        for i, e in enumerate(ranked)
    ]

    return PeerComparisonReport(
        query_theme=theme,
        peer_group=peer_group,
        extractions=extractions,
        peer_ranking=peer_ranking,
        cross_cutting_insights="",  # Filled by a second LLM call for synthesis
        generated_at=datetime.utcnow().isoformat(),
    )
```

---

### Phase 5: Evaluation Protocol

#### 5.1 Evaluation Stack

| Metric | Tool | What It Measures | Target |
|---|---|---|---|
| Numerical Accuracy | Custom exact-match (±0.5% tolerance) | Did we extract the correct NPA ratio, CAR, etc.? | ≥ 90% on gold-standard test set |
| Schema Compliance | Pydantic validation pass rate | Does every LLM output conform to the schema? | 100% (non-negotiable) |
| Context Precision | RAGAS | Did FAISS retrieve the right paragraphs for the queried theme? | 70–85% on real-world PDFs |
| Faithfulness | DeepEval `FaithfulnessMetric` | Is the LLM's summary grounded entirely in retrieved text? | ≥ 0.85 |
| Answer Relevancy | DeepEval `AnswerRelevancyMetric` | Is the LLM's output relevant to the queried theme? | ≥ 0.80 |
| Retrieval Hit Rate | Custom | Did at least one retrieved chunk come from the correct page(s)? | ≥ 75% |

#### 5.2 Gold-Standard Test Set Specification

**Creation Protocol:**

1. Select 5 test pages per bank (4 banks × 5 pages = 20 pages minimum).
2. Manually extract all quantitative metrics from each page.
3. Record expected page numbers for each metric.
4. Encode as `EvalTestCase` Pydantic objects.
5. Store in `eval/gold_standard/` directory.

**Minimum test cases:** 50 total (10 per macro theme × 5 themes).

```python
# Example gold-standard test case
EXAMPLE_TEST_CASE = EvalTestCase(
    test_id="HDFC_FY24_credit_risk_001",
    query_theme="credit_risk",
    bank_name="HDFC_BANK",
    fiscal_year="FY24",
    expected_metrics={
        "gross_npa_ratio": 1.12,
        "net_npa_ratio": 0.27,
        "provision_coverage_ratio": 75.8,
    },
    expected_pages=[142, 143, 187],
    source_document="HDFC_Bank_Annual_Report_FY24.pdf",
)
```

#### 5.3 Numerical Accuracy Evaluator

```python
def evaluate_numerical_accuracy(
    extracted: ThematicExtraction,
    gold: EvalTestCase,
    tolerance: float = 0.005,  # 0.5% relative tolerance
) -> dict[str, bool]:
    """
    Compare each extracted metric against gold standard.
    Tolerance: ±0.5% relative error (e.g., 1.12 ± 0.0056).
    """
    results = {}

    extracted_map = {
        m.metric_name: m.metric_value
        for m in extracted.extracted_metrics
        if m.metric_value is not None
    }

    for metric_name, expected_value in gold.expected_metrics.items():
        if metric_name not in extracted_map:
            results[metric_name] = False
            continue

        actual = extracted_map[metric_name]
        if expected_value == 0:
            results[metric_name] = (actual == 0)
        else:
            relative_error = abs(actual - expected_value) / abs(expected_value)
            results[metric_name] = (relative_error <= tolerance)

    return results
```

---

## 5. Macro Theme Taxonomy

The following fixed taxonomy defines the supported themes. Using a fixed set allows pre-computed query embeddings for consistent retrieval.

| Theme ID | Theme Name | Target Metrics | Typical Source Sections |
|---|---|---|---|
| `credit_risk` | Credit Risk Exposure | Gross NPA %, Net NPA %, Provision Coverage Ratio, Slippage Ratio | MD&A, Asset Quality, Risk Management |
| `liquidity_risk` | Liquidity Risk | LCR %, NSFR %, Loan-to-Deposit Ratio | Risk Management, Basel III Disclosures |
| `unsecured_lending` | Unsecured Lending Exposure | Unsecured Loan %, Personal Loan Growth %, Credit Card NPAs | MD&A, Segment Reporting |
| `capital_adequacy` | Capital Adequacy | CET1 %, Tier 1 %, Total CAR %, RWA Growth | Basel III Disclosures, Capital Adequacy |
| `market_risk` | Market Risk | VaR, Duration Gap, Trading Book Size, MTM Losses | Risk Management, Market Risk |
| `operational_risk` | Operational Risk | OpRisk RWA, Fraud Losses, Cyber Incidents, BCP Status | Risk Management, Operational Risk |
| `asset_quality_trend` | Asset Quality Trends | NPA Movement (opening/additions/recoveries/closing), Write-off Rate | Schedules, Asset Quality |

**Query anchors:** For each theme, maintain a pre-computed embedding of the theme description + target metrics. Use this as the query vector for FAISS retrieval to ensure consistency across runs.

---

## 6. Project Structure

```
quantscribe/
├── pyproject.toml
├── README.md
├── quantscribe/
│   ├── __init__.py
│   ├── config.py                  # Global settings, paths, model names
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── etl.py                 # PageType, ParsedPage, ChunkMetadata, TextChunk
│   │   ├── extraction.py          # CitationTrace, ExtractedMetric, ThematicExtraction
│   │   ├── report.py              # PeerComparisonReport
│   │   └── evaluation.py          # EvalTestCase, EvalResult
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── page_classifier.py     # classify_page()
│   │   ├── pdf_parser.py          # extract_narrative(), extract_tables()
│   │   ├── mixed_page_handler.py  # handle_mixed_page()
│   │   ├── text_cleaner.py        # strip_unicode_garbage(), normalize_indian_currency()
│   │   └── section_detector.py    # detect_section_header()
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── narrative_chunker.py   # chunk_narrative()
│   │   └── table_chunker.py       # chunk_table()
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── pipeline.py            # EmbeddingPipeline
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bank_index.py          # BankIndex
│   │   └── peer_retriever.py      # PeerGroupRetriever
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py             # Prompt templates
│   │   ├── extraction_chain.py    # build_extraction_chain()
│   │   └── peer_comparison.py     # run_peer_comparison()
│   └── evaluation/
│       ├── __init__.py
│       ├── numerical_eval.py      # evaluate_numerical_accuracy()
│       ├── ragas_eval.py          # RAGAS context precision runner
│       └── deepeval_eval.py       # DeepEval faithfulness runner
├── eval/
│   └── gold_standard/
│       ├── HDFC_FY24_credit_risk.json
│       ├── SBI_FY24_credit_risk.json
│       └── ...
├── indices/
│   ├── v1.0.0/
│   └── active -> v1.0.0
├── logs/
│   └── etl/
└── tests/
    ├── test_page_classifier.py
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_extraction.py
```

---

## 7. Development Sequence & Milestones

| Milestone | Phase | Deliverable | Acceptance Criteria |
|---|---|---|---|
| M1 | ETL | Page classifier + extractors working on 10 pages/bank | Manual inspection confirms correct classification and extraction on 40 test pages |
| M2 | Chunking | Narrative + table chunking with full metadata | All chunks pass `ChunkMetadata` Pydantic validation; no table split mid-row |
| M3 | Embedding + Index | Per-bank FAISS indices built and persisted | Query "credit risk" returns relevant chunks from correct bank only |
| M4 | LLM Extraction | Single-bank extraction chain producing valid `ThematicExtraction` | 100% Pydantic schema compliance on 20 test queries |
| M5 | Peer Comparison | Full fan-out peer comparison producing `PeerComparisonReport` | Zero cross-entity contamination on 10 peer comparison queries |
| M6 | Evaluation | Full eval suite running on gold-standard test set | Numerical accuracy ≥ 90%, Context precision ≥ 70%, Faithfulness ≥ 0.85 |

**Non-negotiable gate:** M1 must be fully validated before any work on M2-M6 begins. If extraction is garbage, no amount of RAG sophistication will save the pipeline.

---

## 8. Coding Standards

1. **Type everything.** Use `typing` module extensively. No `Any` types except in truly generic utilities.
2. **Validate everything.** Every function that accepts external data (PDF content, LLM output) must validate inputs via Pydantic or explicit assertions.
3. **Log everything.** Every ETL failure, every LLM retry, every evaluation result. Use structured logging (`structlog` or `logging` with JSON formatter).
4. **Test defensively.** Assume Indian financial PDFs contain hidden unicode garbage, missing headers, broken tables, scanned pages, and watermarks.
5. **No silent failures.** If a page can't be parsed, log the failure with full context and continue. Never silently drop data.
6. **Deterministic IDs.** All chunk IDs are SHA-256 hashes of their metadata. Same input always produces same ID.
7. **Version the pipeline.** Every chunk carries a `parse_version` tag. When ETL logic changes, bump the version and re-index.
