"""
Retrieval & LLM output schemas.

These models define the structured output the LLM MUST produce.
The LangChain PydanticOutputParser enforces these schemas at runtime.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CitationTrace(BaseModel):
    """
    Every extracted metric MUST have one of these.
    This is what makes the system auditable.
    """

    chunk_id: str = Field(description="ID of the source chunk from FAISS retrieval")
    bank_name: str
    document_type: str
    fiscal_year: str
    page_number: int = Field(ge=1)
    section_header: Optional[str] = None
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from FAISS retrieval",
    )
    source_excerpt: str = Field(
        max_length=500,
        description="The exact text span from the chunk that supports this metric",
    )


class ExtractedMetric(BaseModel):
    """
    A single quantitative or qualitative metric extracted by the LLM.

    At least one of metric_value or qualitative_value must be set.
    """

    metric_name: str = Field(
        examples=["gross_npa_ratio", "capital_adequacy_ratio", "lcr_percent"],
    )
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = Field(
        default=None,
        examples=["%", "INR_crore", "basis_points", "ratio"],
    )
    qualitative_value: Optional[str] = Field(
        default=None,
        description=(
            "For non-numeric assessments: "
            "'high', 'moderate', 'low', 'stable', 'deteriorating', 'not_disclosed'"
        ),
    )
    confidence: Literal["high", "medium", "low"]
    citation: CitationTrace

    @model_validator(mode="after")
    def at_least_one_value(self) -> "ExtractedMetric":
        """Ensure at least one of metric_value or qualitative_value is set."""
        if self.metric_value is None and self.qualitative_value is None:
            raise ValueError(
                f"Metric '{self.metric_name}' must have either metric_value or qualitative_value"
            )
        return self


class ThematicExtraction(BaseModel):
    """
    LLM output for a single bank on a single macro theme.
    This is the atomic unit of analysis.

    The LangChain extraction chain MUST produce one of these per bank per query.
    """

    bank_name: str
    fiscal_year: str
    theme: str = Field(
        description="The macro theme being analyzed",
        examples=["credit_risk", "liquidity_risk", "unsecured_lending"],
    )
    risk_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Computed risk score (0=lowest risk, 10=highest risk)",
    )
    risk_rating: Literal["very_low", "low", "moderate", "high", "critical"]
    summary: str = Field(
        max_length=1000,
        description="Qualitative summary grounded entirely in retrieved text",
    )
    extracted_metrics: list[ExtractedMetric] = Field(min_length=1)
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment of disclosure language (-1=very negative, +1=very positive)",
    )
