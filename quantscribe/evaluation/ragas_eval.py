"""
RAGAS evaluation module.

Measures retrieval quality (context precision) and LLM output
groundedness (faithfulness) using RAGAS framework with Gemini as judge.

Usage:
    from quantscribe.evaluation.ragas_eval import run_ragas_evaluation

    results = run_ragas_evaluation(
        theme="credit_risk",
        bank_name="HDFC_BANK",
        query="credit risk gross NPA net NPA provision coverage",
        retrieved_contexts=["chunk1 text...", "chunk2 text..."],
        llm_response="HDFC Bank's Gross NPA ratio stood at 1.33%...",
    )
    print(results)
    # {"context_precision": 0.85, "faithfulness": 0.92}
"""

from __future__ import annotations

from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.evaluation.ragas")


def run_ragas_evaluation(
    theme: str,
    bank_name: str,
    query: str,
    retrieved_contexts: list[str],
    llm_response: str,
) -> dict[str, float]:
    """
    Run RAGAS context precision and faithfulness evaluation.

    Uses Gemini as the LLM judge via LangchainLLMWrapper.

    Args:
        theme: The macro theme queried.
        bank_name: Bank being evaluated.
        query: The retrieval query text.
        retrieved_contexts: List of chunk texts that were sent to the LLM.
        llm_response: The LLM's output (summary + extracted metrics as text).

    Returns:
        Dict with "context_precision" and "faithfulness" scores (0.0 to 1.0).
        Returns -1.0 for any metric that fails to compute.
    """
    settings = get_settings()

    try:
        from ragas import SingleTurnSample, evaluate
        from ragas.metrics import (
            LLMContextPrecisionWithoutReference,
            Faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        logger.error("ragas_import_failed", error=str(e))
        return {"context_precision": -1.0, "faithfulness": -1.0}

    # ── Setup Gemini as RAGAS judge ──
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )
    ragas_llm = LangchainLLMWrapper(llm)

    # ── Build RAGAS sample ──
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        response=llm_response,
    )

    results: dict[str, float] = {}

    # ── Context Precision ──
    try:
        precision_metric = LLMContextPrecisionWithoutReference(llm=ragas_llm)
        precision_score = precision_metric.single_turn_score(sample)
        results["context_precision"] = round(float(precision_score), 4)
        logger.info(
            "ragas_context_precision",
            bank=bank_name,
            theme=theme,
            score=results["context_precision"],
        )
    except Exception as e:
        logger.error("ragas_context_precision_failed", error=str(e)[:200])
        results["context_precision"] = -1.0

    # ── Faithfulness ──
    try:
        faithfulness_metric = Faithfulness(llm=ragas_llm)
        faithfulness_score = faithfulness_metric.single_turn_score(sample)
        results["faithfulness"] = round(float(faithfulness_score), 4)
        logger.info(
            "ragas_faithfulness",
            bank=bank_name,
            theme=theme,
            score=results["faithfulness"],
        )
    except Exception as e:
        logger.error("ragas_faithfulness_failed", error=str(e)[:200])
        results["faithfulness"] = -1.0

    return results


def run_ragas_batch(
    evaluations: list[dict],
) -> list[dict]:
    """
    Run RAGAS evaluation on a batch of extractions.

    Args:
        evaluations: List of dicts, each with keys:
            - theme, bank_name, query, retrieved_contexts, llm_response

    Returns:
        List of result dicts with RAGAS scores appended.
    """
    all_results = []

    for i, eval_input in enumerate(evaluations):
        logger.info(
            "ragas_batch_progress",
            index=i + 1,
            total=len(evaluations),
            bank=eval_input["bank_name"],
        )

        scores = run_ragas_evaluation(
            theme=eval_input["theme"],
            bank_name=eval_input["bank_name"],
            query=eval_input["query"],
            retrieved_contexts=eval_input["retrieved_contexts"],
            llm_response=eval_input["llm_response"],
        )

        result = {**eval_input, **scores}
        all_results.append(result)

    return all_results
    