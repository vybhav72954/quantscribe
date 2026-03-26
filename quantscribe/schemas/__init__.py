"""
QuantScribe Pydantic schemas — the system contract.

Every phase of the pipeline produces or consumes these models.
Import from here, never define schemas inline in other modules.
"""

from quantscribe.schemas.etl import (
    PageType,
    ChunkMetadata,
    ParsedPage,
    TextChunk,
)
from quantscribe.schemas.extraction import (
    CitationTrace,
    ExtractedMetric,
    ThematicExtraction,
)
from quantscribe.schemas.report import PeerComparisonReport
from quantscribe.schemas.evaluation import EvalTestCase, EvalResult

__all__ = [
    "PageType",
    "ChunkMetadata",
    "ParsedPage",
    "TextChunk",
    "CitationTrace",
    "ExtractedMetric",
    "ThematicExtraction",
    "PeerComparisonReport",
    "EvalTestCase",
    "EvalResult",
]
