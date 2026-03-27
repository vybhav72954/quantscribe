"""
LLM extraction chain with Pydantic output parsing and retry logic.

Uses LangChain + Gemini API to extract structured ThematicExtraction
objects from retrieved financial text.

IMPORTANT: Uses Gemini's native structured output (with_structured_output)
instead of PydanticOutputParser. This bypasses the markdown fence issue
where Gemini wraps JSON in ```json blocks that PydanticOutputParser
cannot parse, causing silent failures after all retries are exhausted.

Usage:
    from quantscribe.llm.extraction_chain import build_extraction_chain

    chain = build_extraction_chain(max_retries=3)
    result = chain({"theme": "credit_risk", "bank_contexts": "..."})
    # result is a validated ThematicExtraction object
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from quantscribe.schemas.extraction import ThematicExtraction
from quantscribe.llm.prompts import THEMATIC_EXTRACTION_PROMPT_STRUCTURED
from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.llm.extraction_chain")


def build_extraction_chain(max_retries: int = 3):
    """
    Build a LangChain extraction chain that returns a callable.

    The callable takes dict(theme, bank_contexts) and returns
    a validated ThematicExtraction object.

    Args:
        max_retries: Number of retry attempts on malformed JSON or
                     failed citation validation.

    Returns:
        A callable: dict -> ThematicExtraction
    """
    settings = get_settings()

    # ── LLM setup ──
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
        max_output_tokens=settings.llm_max_output_tokens,
    )

    # ── Use Gemini native structured output ──
    # This forces Gemini to return valid JSON matching the schema directly,
    # bypassing the markdown fence wrapping that breaks PydanticOutputParser.
    structured_llm = llm.with_structured_output(ThematicExtraction)

    # ── Prompt template (no {format_instructions} needed) ──
    prompt = PromptTemplate(
        template=THEMATIC_EXTRACTION_PROMPT_STRUCTURED,
        input_variables=["theme", "bank_contexts"],
    )

    # ── Chain: prompt → structured LLM ──
    chain = prompt | structured_llm

    logger.info(
        "extraction_chain_built",
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_retries=max_retries,
        output_mode="structured_output",
    )

    # ── Callable with retry + validation ──
    def invoke_with_retry(inputs: dict) -> ThematicExtraction:
        """
        Invoke the extraction chain with retry logic.

        On failure:
        1. Logs the full error message (visible in logs/quantscribe.jsonl)
        2. Appends the error to context so the LLM can self-correct
        3. Retries up to max_retries times

        After successful parse:
        - Validates that every citation excerpt appears in the provided context
        """
        working_inputs = {
            "theme": inputs["theme"],
            "bank_contexts": inputs["bank_contexts"],
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                result = chain.invoke(working_inputs)

                # Gemini structured output can return None on schema mismatch
                if result is None:
                    raise ValueError(
                        "Gemini returned None — schema mismatch or empty response. "
                        "Check that all required fields are populated."
                    )

                # ── Post-validation: check citations ──
                _validate_citations(result, inputs["bank_contexts"])

                logger.info(
                    "extraction_success",
                    theme=inputs["theme"],
                    bank=result.bank_name,
                    metrics=len(result.extracted_metrics),
                    risk_score=result.risk_score,
                    attempt=attempt + 1,
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "extraction_attempt_failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e)[:500],
                    theme=inputs["theme"],
                )

                if attempt < max_retries - 1:
                    working_inputs["bank_contexts"] += (
                        f"\n\n[SYSTEM: Your previous response produced an error: "
                        f"{str(e)[:200]}. "
                        f"Please fix the output and try again. "
                        f"Ensure all required fields are present and valid.]"
                    )

        logger.error(
            "extraction_failed",
            theme=inputs["theme"],
            retries=max_retries,
            error=str(last_error)[:500],
        )
        raise RuntimeError(
            f"Extraction failed for theme='{inputs.get('theme')}' "
            f"after {max_retries} retries: {last_error}"
        ) from last_error

    return invoke_with_retry


def _validate_citations(
    extraction: ThematicExtraction,
    context_text: str,
    min_overlap: float = 0.5,
) -> None:
    """
    Validate that every citation's source_excerpt actually appears
    in the provided context.

    Uses word-level overlap since LLMs sometimes paraphrase slightly.
    Threshold is 0.5 (50%) to account for Gemini's tendency to lightly
    rephrase excerpts even in structured output mode.

    Args:
        extraction: The parsed ThematicExtraction object.
        context_text: The original bank_contexts string sent to the LLM.
        min_overlap: Minimum fraction of excerpt words that must appear
                     in the context (default: 50%).

    Raises:
        ValueError: If any citation fails validation.
    """
    context_words = set(context_text.lower().split())

    for metric in extraction.extracted_metrics:
        excerpt = metric.citation.source_excerpt

        # Fast path: exact substring match
        if excerpt in context_text:
            continue

        # Slow path: word-level overlap check
        excerpt_words = set(excerpt.lower().split())
        if not excerpt_words:
            continue

        overlap = len(excerpt_words & context_words) / len(excerpt_words)

        if overlap < min_overlap:
            raise ValueError(
                f"Citation validation failed for metric '{metric.metric_name}': "
                f"only {overlap:.0%} word overlap (minimum {min_overlap:.0%}). "
                f"Excerpt: '{excerpt[:150]}'"
            )

    logger.info(
        "citations_validated",
        bank=extraction.bank_name,
        metrics_checked=len(extraction.extracted_metrics),
    )
