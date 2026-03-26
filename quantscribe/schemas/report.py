"""
Peer comparison report schema.

This is the final output of the entire pipeline —
one PeerComparisonReport per user query.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quantscribe.schemas.extraction import ThematicExtraction


class PeerRankEntry(BaseModel):
    """A single entry in the peer ranking."""

    bank: str
    risk_score: float = Field(ge=0.0, le=10.0)
    rank: int = Field(ge=1)


class PeerComparisonReport(BaseModel):
    """
    The final cross-bank comparison output.
    One of these is generated per query.
    """

    query_theme: str
    peer_group: list[str] = Field(
        min_length=2,
        description="List of bank names being compared",
    )
    extractions: list[ThematicExtraction]
    peer_ranking: list[PeerRankEntry] = Field(
        description="Banks ranked by risk_score for this theme, ascending",
    )
    cross_cutting_insights: str = Field(
        max_length=2000,
        description="Synthesized insights comparing trends across the peer group",
    )
    generated_at: str = Field(description="ISO 8601 timestamp")
