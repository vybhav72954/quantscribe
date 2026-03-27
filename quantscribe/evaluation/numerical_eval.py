"""
Numerical accuracy evaluator.

Compares extracted metrics against gold-standard test cases
with configurable relative tolerance.

Two-layer matching strategy:
1. Exact match on metric_name (fast path)
2. Fuzzy match via alias mapping + keyword overlap (safety net)

This ensures evaluation works even if the LLM uses slightly different
metric names (e.g. "gnpa_ratio" vs "gross_npa_ratio").
"""

from __future__ import annotations

from quantscribe.schemas.extraction import ThematicExtraction
from quantscribe.schemas.evaluation import EvalTestCase
from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.evaluation.numerical")

# ── Alias mapping: common LLM variations → gold standard names ──
# Keys are lowercase. If Gemini uses any of these, they map to the gold name.
METRIC_ALIASES: dict[str, str] = {
    # credit_risk
    "gnpa_ratio": "gross_npa_ratio",
    "gnpa": "gross_npa_ratio",
    "gross_npa": "gross_npa_ratio",
    "gross_npa_percent": "gross_npa_ratio",
    "gross_npa_percentage": "gross_npa_ratio",
    "nnpa_ratio": "net_npa_ratio",
    "nnpa": "net_npa_ratio",
    "net_npa": "net_npa_ratio",
    "net_npa_percent": "net_npa_ratio",
    "net_npa_percentage": "net_npa_ratio",
    "pcr": "provision_coverage_ratio",
    "provision_coverage": "provision_coverage_ratio",
    "provision_coverage_percent": "provision_coverage_ratio",
    # capital_adequacy
    "cet1": "cet1_ratio",
    "cet_1_ratio": "cet1_ratio",
    "cet_1": "cet1_ratio",
    "common_equity_tier1": "cet1_ratio",
    "tier1": "tier1_ratio",
    "tier_1_ratio": "tier1_ratio",
    "tier_1": "tier1_ratio",
    "car": "total_car",
    "crar": "total_car",
    "capital_adequacy_ratio": "total_car",
    "total_crar": "total_car",
    "total_capital_ratio": "total_car",
    # liquidity_risk
    "lcr": "lcr_percent",
    "liquidity_coverage_ratio": "lcr_percent",
    "nsfr": "nsfr_percent",
    "net_stable_funding_ratio": "nsfr_percent",
    "credit_deposit_ratio": "loan_to_deposit_ratio",
    "cd_ratio": "loan_to_deposit_ratio",
    # unsecured_lending
    "personal_loan_pct": "personal_loan_percent",
    "personal_loans_percent": "personal_loan_percent",
    "credit_card_npa_ratio": "credit_card_npa",
    "cc_npa": "credit_card_npa",
}

# ── Keyword sets for last-resort fuzzy matching ──
# If alias lookup fails, check if the extracted metric name contains
# enough keywords to match a gold standard metric.
METRIC_KEYWORDS: dict[str, set[str]] = {
    "gross_npa_ratio": {"gross", "npa", "gnpa", "gnpas", "nonperforming"},
    "net_npa_ratio": {"net", "npa", "nnpa", "nnpas", "nonperforming"},
    "provision_coverage_ratio": {"provision", "coverage", "pcr"},
    "slippage_ratio": {"slippage"},
    "cet1_ratio": {"cet1", "common", "equity"},
    "tier1_ratio": {"tier1", "tier"},
    "total_car": {"car", "crar", "capital", "adequacy"},
    "lcr_percent": {"lcr", "liquidity", "coverage"},
    "nsfr_percent": {"nsfr", "stable", "funding"},
    "personal_loan_percent": {"personal", "loan"},
    "credit_card_npa": {"credit", "card"},
    "loan_to_deposit_ratio": {"loan", "deposit"},
}


def evaluate_numerical_accuracy(
    extracted: ThematicExtraction,
    gold: EvalTestCase,
    tolerance: float | None = None,
) -> dict[str, bool]:
    """
    Compare each extracted metric against gold standard.

    Uses a two-layer matching strategy:
    1. Exact match on metric_name
    2. Alias mapping + keyword fuzzy match

    Args:
        extracted: LLM extraction output.
        gold: Ground-truth test case.
        tolerance: Relative tolerance (default: 0.5% from config).

    Returns:
        Dict mapping metric_name -> bool (True if within tolerance).
    """
    settings = get_settings()
    tol = tolerance or settings.numerical_tolerance

    # Build lookup from extracted metrics (exact name → value)
    extracted_map: dict[str, float | None] = {
        m.metric_name: m.metric_value
        for m in extracted.extracted_metrics
    }

    results: dict[str, bool] = {}

    for gold_name, expected_value in gold.expected_metrics.items():
        # ── Layer 1: Exact match ──
        actual = extracted_map.get(gold_name)

        # ── Layer 2: Alias + fuzzy match ──
        if actual is None:
            actual = _fuzzy_lookup(gold_name, extracted_map)

        if actual is None:
            logger.warn(
                "metric_not_extracted",
                metric=gold_name,
                bank=gold.bank_name,
                test_id=gold.test_id,
                available_metrics=list(extracted_map.keys()),
            )
            results[gold_name] = False
            continue

        # ── Compare values ──
        if expected_value == 0:
            match = actual == 0
        else:
            relative_error = abs(actual - expected_value) / abs(expected_value)
            match = relative_error <= tol

        results[gold_name] = match

        if not match:
            logger.warn(
                "numerical_mismatch",
                metric=gold_name,
                expected=expected_value,
                actual=actual,
                test_id=gold.test_id,
            )
        else:
            logger.info(
                "metric_matched",
                metric=gold_name,
                expected=expected_value,
                actual=actual,
                test_id=gold.test_id,
            )

    accuracy = sum(results.values()) / max(len(results), 1)
    logger.info(
        "numerical_accuracy",
        test_id=gold.test_id,
        accuracy=accuracy,
        total_metrics=len(results),
        passed=sum(results.values()),
    )

    return results


def _fuzzy_lookup(
    gold_name: str,
    extracted_map: dict[str, float | None],
) -> float | None:
    """
    Try to find a matching metric value using alias mapping
    and keyword overlap.

    Args:
        gold_name: The expected metric name from gold standard.
        extracted_map: Dict of extracted metric_name → value.

    Returns:
        The matched metric value, or None if no match found.
    """
    # ── Strategy A: Check if any extracted name is an alias for gold_name ──
    for extracted_name, value in extracted_map.items():
        if value is None:
            continue
        canonical = METRIC_ALIASES.get(extracted_name.lower())
        if canonical == gold_name:
            logger.info(
                "metric_alias_match",
                gold=gold_name,
                extracted=extracted_name,
                via="alias",
            )
            return value

    # ── Strategy B: Keyword overlap ──
    keywords = METRIC_KEYWORDS.get(gold_name)
    if not keywords:
        return None

    best_match: tuple[str, float | None, int] | None = None  # (name, value, overlap_count)

    for extracted_name, value in extracted_map.items():
        if value is None:
            continue
        # Tokenize the extracted name — strip punctuation so "(GNPAs)" → "gnpas"
        import re
        cleaned_name = re.sub(r"[^a-zA-Z0-9\s_]", "", extracted_name.lower())
        name_tokens = set(cleaned_name.replace("_", " ").split())
        overlap = len(keywords & name_tokens)

        # Require at least 2 keyword matches (or 1 if the keyword set is small)
        min_required = min(2, len(keywords))
        if overlap >= min_required:
            if best_match is None or overlap > best_match[2]:
                best_match = (extracted_name, value, overlap)

    if best_match is not None:
        logger.info(
            "metric_fuzzy_match",
            gold=gold_name,
            extracted=best_match[0],
            overlap=best_match[2],
            via="keyword",
        )
        return best_match[1]

    return None
    