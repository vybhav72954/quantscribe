"""
DeepEval evaluation module.

Measures LLM output faithfulness and answer relevancy
using DeepEval framework with Gemini as the judge LLM.

Usage:
    from quantscribe.evaluation.deepeval_eval import run_deepeval_evaluation

    results = run_deepeval_evaluation(
        theme="credit_risk",
        bank_name="HDFC_BANK",
        query="credit risk gross NPA net NPA provision coverage",
        retrieved_contexts=["chunk1 text...", "chunk2 text..."],
        llm_response="HDFC Bank's Gross NPA ratio stood at 1.33%...",
    )
    print(results)
    # {"faithfulness": 0.92, "answer_relevancy": 0.88}
"""

from __future__ import annotations

from typing import Optional

from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.evaluation.deepeval")


class GeminiDeepEvalModel:
    """
    Wraps Gemini for use as DeepEval's judge LLM.

    DeepEval defaults to OpenAI. This class provides a Gemini-backed
    alternative by implementing the interface DeepEval expects.
    """

    def __init__(self, model_name: str | None = None):
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = get_settings()
        self.model_name = model_name or settings.llm_model

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.0,
            google_api_key=settings.google_api_key,
        )

    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini."""
        response = self.llm.invoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return self.model_name


def run_deepeval_evaluation(
    theme: str,
    bank_name: str,
    query: str,
    retrieved_contexts: list[str],
    llm_response: str,
) -> dict[str, float]:
    """
    Run DeepEval faithfulness and answer relevancy evaluation.

    Uses Gemini as the judge LLM.

    Args:
        theme: The macro theme queried.
        bank_name: Bank being evaluated.
        query: The retrieval query text.
        retrieved_contexts: List of chunk texts sent to the LLM.
        llm_response: The LLM's output text.

    Returns:
        Dict with "faithfulness" and "answer_relevancy" scores (0.0 to 1.0).
        Returns -1.0 for any metric that fails to compute.
    """
    try:
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        logger.error("deepeval_import_failed", error=str(e))
        return {"faithfulness": -1.0, "answer_relevancy": -1.0}

    # ── Build test case ──
    test_case = LLMTestCase(
        input=query,
        actual_output=llm_response,
        retrieval_context=retrieved_contexts,
    )

    results: dict[str, float] = {}
    settings = get_settings()

    # ── Faithfulness ──
    try:
        faithfulness = FaithfulnessMetric(
            threshold=0.85,
            model=settings.llm_model,
        )
        faithfulness.measure(test_case)
        results["faithfulness"] = round(float(faithfulness.score), 4)
        logger.info(
            "deepeval_faithfulness",
            bank=bank_name,
            theme=theme,
            score=results["faithfulness"],
            reason=faithfulness.reason[:200] if faithfulness.reason else "N/A",
        )
    except Exception as e:
        logger.error(
            "deepeval_faithfulness_failed",
            error=str(e)[:300],
            bank=bank_name,
        )
        results["faithfulness"] = -1.0

    # ── Answer Relevancy ──
    try:
        relevancy = AnswerRelevancyMetric(
            threshold=0.80,
            model=settings.llm_model,
        )
        relevancy.measure(test_case)
        results["answer_relevancy"] = round(float(relevancy.score), 4)
        logger.info(
            "deepeval_answer_relevancy",
            bank=bank_name,
            theme=theme,
            score=results["answer_relevancy"],
            reason=relevancy.reason[:200] if relevancy.reason else "N/A",
        )
    except Exception as e:
        logger.error(
            "deepeval_answer_relevancy_failed",
            error=str(e)[:300],
            bank=bank_name,
        )
        results["answer_relevancy"] = -1.0

    return results


def run_deepeval_batch(
    evaluations: list[dict],
) -> list[dict]:
    """
    Run DeepEval evaluation on a batch of extractions.

    Args:
        evaluations: List of dicts, each with keys:
            - theme, bank_name, query, retrieved_contexts, llm_response

    Returns:
        List of result dicts with DeepEval scores appended.
    """
    all_results = []

    for i, eval_input in enumerate(evaluations):
        logger.info(
            "deepeval_batch_progress",
            index=i + 1,
            total=len(evaluations),
            bank=eval_input["bank_name"],
        )

        scores = run_deepeval_evaluation(
            theme=eval_input["theme"],
            bank_name=eval_input["bank_name"],
            query=eval_input["query"],
            retrieved_contexts=eval_input["retrieved_contexts"],
            llm_response=eval_input["llm_response"],
        )

        result = {**eval_input, **scores}
        all_results.append(result)

    return all_results
