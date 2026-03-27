"""
Peer comparison orchestrator.

End-to-end pipeline:
1. Embed the thematic query using pre-defined query anchors
2. Fan-out retrieval to each bank's FAISS index independently
3. Format retrieved chunks with rigid delimiters
4. Run LLM extraction ONCE PER BANK (prevents cross-entity contamination)
5. Back-fill relevance_score on each CitationTrace from actual FAISS scores
6. Rank banks by risk_score
7. Synthesize cross-cutting insights via a second LLM call
8. Return a validated PeerComparisonReport

Usage:
    from quantscribe.llm.peer_comparison import run_peer_comparison

    report = run_peer_comparison(
        theme="credit_risk",
        peer_group=["HDFC_BANK", "SBI"],
        retriever=retriever,
        embedding_pipeline=embedder,
        extraction_chain=chain,
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI

from quantscribe.schemas.extraction import ThematicExtraction
from quantscribe.schemas.report import PeerComparisonReport, PeerRankEntry
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever
from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.llm.prompts import (
    BANK_CONTEXT_TEMPLATE,
    CHUNK_TEMPLATE,
    PEER_SYNTHESIS_PROMPT,
)
from quantscribe.config import get_settings, THEME_TAXONOMY
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.llm.peer_comparison")


def run_peer_comparison(
    theme: str,
    peer_group: list[str],
    retriever: PeerGroupRetriever,
    embedding_pipeline: EmbeddingPipeline,
    extraction_chain,
    top_k_per_bank: int = 5,
) -> PeerComparisonReport:
    """
    Run a full peer comparison for a given theme.

    Args:
        theme:              Macro theme ID (e.g., "credit_risk").
        peer_group:         List of bank names to compare.
        retriever:          PeerGroupRetriever with loaded FAISS indices.
        embedding_pipeline: EmbeddingPipeline for query embedding.
        extraction_chain:   Callable from build_extraction_chain().
        top_k_per_bank:     Number of chunks to retrieve per bank.

    Returns:
        Validated PeerComparisonReport with extractions, rankings, and insights.
    """
    logger.info(
        "peer_comparison_start",
        theme=theme,
        peer_group=peer_group,
        top_k=top_k_per_bank,
    )

    # ── Step 1: Embed thematic query ──
    query_text = _build_query_text(theme)
    query_vector = embedding_pipeline.embed_query(query_text)

    logger.info("query_embedded", theme=theme, query=query_text[:80])

    # ── Step 2: Fan-out retrieval ──
    all_results = retriever.retrieve(
        query_vector, peer_group, top_k_per_bank=top_k_per_bank,
    )

    logger.info(
        "retrieval_complete",
        banks_with_results=list(all_results.keys()),
        chunks_per_bank={b: len(r) for b, r in all_results.items()},
    )

    # ── Step 3: Format contexts + run extraction per bank ──
    extractions: list[ThematicExtraction] = []

    for bank_name in peer_group:
        if bank_name not in all_results or not all_results[bank_name]:
            logger.warning("no_chunks_retrieved", bank=bank_name, theme=theme)
            continue

        results = all_results[bank_name]

        # Build a page → score lookup for back-filling relevance_score
        # after extraction (the LLM doesn't know FAISS scores)
        page_score_map: dict[int, float] = {
            r["metadata"]["page_number"]: r["score"]
            for r in results
        }

        # Format chunks with rigid delimiters
        bank_context = _format_bank_context(bank_name, results)

        logger.info(
            "extraction_start",
            bank=bank_name,
            chunks=len(results),
            context_chars=len(bank_context),
        )

        # ── Step 4: Call extraction chain (ONE call per bank) ──
        try:
            extraction = extraction_chain({
                "theme": theme,
                "bank_contexts": bank_context,
            })

            # ── Step 5: Back-fill relevance_score from real FAISS scores ──
            # The LLM leaves relevance_score=0.0 (the default).
            # We fill it with the actual cosine similarity for the cited page.
            # If the page isn't in our retrieval results, use the minimum score.
            min_score = min(r["score"] for r in results)
            for metric in extraction.extracted_metrics:
                cited_page = metric.citation.page_number
                metric.citation.relevance_score = page_score_map.get(
                    cited_page, min_score
                )

            extractions.append(extraction)

            logger.info(
                "extraction_complete",
                bank=bank_name,
                risk_score=extraction.risk_score,
                metrics=len(extraction.extracted_metrics),
            )
        except Exception as e:
            logger.error(
                "extraction_failed_for_bank",
                bank=bank_name,
                theme=theme,
                error=str(e)[:300],
            )
            # Continue with other banks — don't abort the whole comparison

    if not extractions:
        raise RuntimeError(
            f"No successful extractions for theme='{theme}'. "
            f"All banks failed. Check logs for details."
        )

    # ── Step 6: Rank by risk score (ascending = lowest risk first) ──
    ranked = sorted(extractions, key=lambda x: x.risk_score)
    peer_ranking = [
        PeerRankEntry(bank=e.bank_name, risk_score=e.risk_score, rank=i + 1)
        for i, e in enumerate(ranked)
    ]

    # ── Step 7: Synthesize cross-cutting insights ──
    insights = _synthesize_insights(theme, peer_group, extractions)

    # ── Step 8: Build report ──
    report = PeerComparisonReport(
        query_theme=theme,
        peer_group=peer_group,
        extractions=extractions,
        peer_ranking=peer_ranking,
        cross_cutting_insights=insights,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "peer_comparison_complete",
        theme=theme,
        banks_compared=len(extractions),
        top_bank=peer_ranking[0].bank if peer_ranking else "N/A",
    )

    return report


# ── Internal helpers ──


def _build_query_text(theme: str) -> str:
    """
    Build the query text for embedding.

    Uses the pre-defined query_anchor from THEME_TAXONOMY if available,
    otherwise constructs a generic query from the theme name.
    """
    if theme in THEME_TAXONOMY:
        return THEME_TAXONOMY[theme]["query_anchor"]

    readable = theme.replace("_", " ")
    return f"{readable} risk exposure analysis financial metrics"


def _format_bank_context(bank_name: str, results: list[dict]) -> str:
    """
    Format retrieved chunks for a single bank using rigid delimiters.

    Each chunk is wrapped with page number and section header.
    The entire bank's context is wrapped in BEGIN/END markers.

    Requires that metadata_store entries contain a "content" key —
    guaranteed by BankIndex.add() storing TextChunk objects.
    """
    chunks_formatted: list[str] = []

    for r in results:
        meta = r["metadata"]
        content = meta.get("content", "")

        if not content:
            logger.error(
                "chunk_content_missing_in_metadata",
                bank=bank_name,
                chunk_id=meta.get("chunk_id", "unknown"),
                page=meta.get("page_number", "unknown"),
                hint=(
                    "Re-run kaggle_embed.py to rebuild indices with content storage. "
                    "Old indices built without content cannot be used for LLM extraction."
                ),
            )
            continue

        chunk_text = CHUNK_TEMPLATE.format(
            page_number=meta.get("page_number", "N/A"),
            section_header=meta.get("section_header", "N/A"),
            content=content,
        )
        chunks_formatted.append(chunk_text)

    if not chunks_formatted:
        raise RuntimeError(
            f"All retrieved chunks for {bank_name} are missing content. "
            f"Rebuild the FAISS indices using the updated kaggle_embed.py."
        )

    chunks_joined = "\n\n---\n\n".join(chunks_formatted)

    first_meta = results[0]["metadata"]
    fiscal_year = first_meta.get("fiscal_year", "N/A")
    document_type = first_meta.get("document_type", "N/A")

    return BANK_CONTEXT_TEMPLATE.format(
        bank_name=bank_name,
        fiscal_year=fiscal_year,
        document_type=document_type,
        chunks=chunks_joined,
    )


def _synthesize_insights(
    theme: str,
    peer_group: list[str],
    extractions: list[ThematicExtraction],
) -> str:
    """
    Generate cross-cutting insights by making a second LLM call.
    """
    settings = get_settings()

    extractions_summary = []
    for ext in extractions:
        summary = {
            "bank": ext.bank_name,
            "risk_score": ext.risk_score,
            "risk_rating": ext.risk_rating,
            "metrics": {
                m.metric_name: {
                    "value": m.metric_value,
                    "unit": m.metric_unit,
                    "confidence": m.confidence,
                }
                for m in ext.extracted_metrics
            },
            "sentiment": ext.sentiment_score,
        }
        extractions_summary.append(summary)

    extractions_json = json.dumps(extractions_summary, indent=2)

    prompt_text = PEER_SYNTHESIS_PROMPT.format(
        theme=theme,
        peer_group=", ".join(peer_group),
        extractions_json=extractions_json,
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
            max_output_tokens=settings.llm_max_output_tokens,
        )

        response = llm.invoke(prompt_text)
        insights = response.content.strip()

        if len(insights) > 2000:
            insights = insights[:1997] + "..."

        logger.info("synthesis_complete", theme=theme, insights_chars=len(insights))
        return insights

    except Exception as exc:
        logger.error("synthesis_failed", theme=theme, error=str(exc)[:200])
        return (
            f"Cross-cutting synthesis unavailable due to LLM error. "
            f"Individual bank extractions are available in the report. "
            f"Banks compared: {', '.join(ext.bank_name for ext in extractions)}. "
            f"Risk scores: {', '.join(f'{ext.bank_name}={ext.risk_score}' for ext in extractions)}."
        )
        