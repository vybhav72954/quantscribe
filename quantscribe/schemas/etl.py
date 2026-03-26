"""
ETL & Chunking schemas.

These models govern the output of Phase 1 (ETL) and Phase 2 (Chunking).
Every chunk that enters the embedding pipeline MUST be a valid TextChunk.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PageType(str, Enum):
    """Classification label for a single PDF page."""

    NARRATIVE = "narrative"
    TABULAR = "tabular"
    MIXED = "mixed"
    GRAPHICAL = "graphical"


class ChunkMetadata(BaseModel):
    """
    Metadata envelope attached to EVERY chunk before embedding.

    This is the primary mechanism preventing cross-entity contamination.
    Every field here is queryable and auditable.
    """

    chunk_id: str = Field(
        description=(
            "Deterministic hash: "
            "sha256(bank_name + doc_type + fiscal_year + page_number + chunk_index)"
        )
    )
    bank_name: str = Field(
        description="Standardized bank identifier (uppercase snake_case)",
        examples=["HDFC_BANK", "SBI", "ICICI_BANK", "AXIS_BANK"],
    )
    document_type: Literal["annual_report", "earnings_call", "investor_presentation"]
    fiscal_year: str = Field(
        pattern=r"^FY\d{2}$",
        description="Fiscal year in FYxx format",
        examples=["FY23", "FY24"],
    )
    page_number: int = Field(ge=1)
    section_header: Optional[str] = Field(
        default=None,
        description="Nearest extractable section header (e.g., 'Management Discussion & Analysis')",
    )
    page_type: PageType
    chunk_index: int = Field(
        ge=0,
        description="Sequential index of this chunk within the page",
    )
    token_count: int = Field(ge=1)
    parse_version: str = Field(
        description="ETL pipeline version that produced this chunk",
        examples=["etl_v1.0.0", "etl_v1.2.3"],
    )

    @field_validator("bank_name")
    @classmethod
    def normalize_bank_name(cls, v: str) -> str:
        """Enforce uppercase snake_case for all bank names."""
        return v.upper().replace(" ", "_").replace("-", "_")

    @staticmethod
    def generate_chunk_id(
        bank_name: str,
        document_type: str,
        fiscal_year: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """Generate a deterministic chunk ID via SHA-256."""
        raw = f"{bank_name}_{document_type}_{fiscal_year}_{page_number}_{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ParsedPage(BaseModel):
    """Output of the page classification + extraction step."""

    page_number: int = Field(ge=1)
    page_type: PageType
    raw_text: Optional[str] = None
    tables: Optional[list[dict]] = Field(
        default=None,
        description="List of tables as list-of-dicts (each dict = one row)",
    )
    extraction_warnings: list[str] = Field(default_factory=list)
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in page classification (0-1)",
    )


class TextChunk(BaseModel):
    """
    A single chunk ready for embedding.

    This is the atomic unit that enters the FAISS index.
    Every TextChunk has full metadata traceability back to its source.
    """

    content: str = Field(min_length=10)
    metadata: ChunkMetadata
    content_type: Literal["narrative", "table_text", "table_structured"]
