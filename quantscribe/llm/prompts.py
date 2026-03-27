"""
Prompt templates for LLM extraction.

These templates use rigid delimiters to prevent cross-entity contamination
in the LLM's context window.

Two variants of the extraction prompt exist:
- THEMATIC_EXTRACTION_PROMPT:            Original version with {format_instructions}.
                                          Used with PydanticOutputParser (legacy).
- THEMATIC_EXTRACTION_PROMPT_STRUCTURED: No format_instructions placeholder.
                                          Used with llm.with_structured_output()
                                          which handles schema enforcement natively.
"""

# ── Legacy prompt (PydanticOutputParser) ──
THEMATIC_EXTRACTION_PROMPT = """You are a quantitative financial analyst specializing in Indian banking.
You are given retrieved text from a bank's annual report. Your task is to extract specific metrics
related to the queried theme and produce a structured risk assessment.

CRITICAL RULES:
1. ONLY use information from the provided context below. Do NOT use any prior knowledge.
2. If a metric is not explicitly stated in the context, set its confidence to "low" and
   qualitative_value to "not_disclosed".
3. Every metric you extract MUST include a citation with the exact source_excerpt
   (max 500 chars) from the context.
4. Risk scores must be between 0.0 (lowest risk) and 10.0 (highest risk).
5. Do NOT infer, calculate, or hallucinate any numbers not present in the text.
6. Sentiment score should reflect the tone of the disclosure language
   (-1.0 = very negative, +1.0 = very positive).

QUERIED THEME: {theme}

{bank_contexts}

{format_instructions}
"""

# ── Structured output prompt (with_structured_output) ──
# No {format_instructions} — Gemini enforces the schema natively.
THEMATIC_EXTRACTION_PROMPT_STRUCTURED = """You are a quantitative financial analyst specializing in Indian banking.
You are given retrieved text from a bank's annual report. Your task is to extract specific metrics
related to the queried theme and produce a structured risk assessment.

CRITICAL RULES:
1. ONLY use information from the provided context below. Do NOT use any prior knowledge.
2. If a metric is not explicitly stated in the context, set its confidence to "low" and
   qualitative_value to "not_disclosed".
3. Every metric you extract MUST include a citation with the exact source_excerpt
   (max 500 chars) from the context.
4. Risk scores must be between 0.0 (lowest risk) and 10.0 (highest risk).
5. Do NOT infer, calculate, or hallucinate any numbers not present in the text.
6. Sentiment score should reflect the tone of the disclosure language
   (-1.0 = very negative, +1.0 = very positive).
7. You MUST always set either metric_value (float) or qualitative_value (string) for every metric.
   Never leave both as null.
8. bank_name must exactly match the bank identifier from the context header (e.g. HDFC_BANK, SBI).
9. fiscal_year must be in FYxx format (e.g. FY25).

METRIC NAMING RULES:
You MUST use the EXACT metric_name values listed below. Do NOT invent your own names.
Do NOT use long descriptive names like "Gross Non-Performing Assets (GNPAs) to Gross Advances".
Use ONLY the short snake_case identifiers provided.

{metric_names_instruction}

If you find a metric in the text that does not match any of the names above, you may include it
using a short snake_case name (e.g. "slippage_ratio", "writeoff_rate"), but ALWAYS prioritize
the standard names listed above.

QUERIED THEME: {theme}

{bank_contexts}
"""

# ── Per-theme metric name instructions ──
# These are injected into the prompt by peer_comparison.py
METRIC_NAMES_BY_THEME: dict[str, str] = {
    "credit_risk": (
        "For credit_risk, use EXACTLY these metric_name values:\n"
        "- gross_npa_ratio (for Gross NPA % or GNPA ratio)\n"
        "- net_npa_ratio (for Net NPA % or NNPA ratio)\n"
        "- provision_coverage_ratio (for PCR, provision coverage)\n"
        "- slippage_ratio (for slippage ratio / fresh slippages to advances)\n"
        "- credit_cost (for credit cost ratio)\n"
    ),
    "capital_adequacy": (
        "For capital_adequacy, use EXACTLY these metric_name values:\n"
        "- cet1_ratio (for CET1 / Common Equity Tier 1 ratio)\n"
        "- tier1_ratio (for Tier 1 capital ratio)\n"
        "- total_car (for Total CAR / CRAR / Capital Adequacy Ratio)\n"
        "- rwa_growth (for Risk Weighted Assets growth rate)\n"
    ),
    "liquidity_risk": (
        "For liquidity_risk, use EXACTLY these metric_name values:\n"
        "- lcr_percent (for LCR / Liquidity Coverage Ratio)\n"
        "- nsfr_percent (for NSFR / Net Stable Funding Ratio)\n"
        "- loan_to_deposit_ratio (for loan-to-deposit / credit-deposit ratio)\n"
    ),
    "unsecured_lending": (
        "For unsecured_lending, use EXACTLY these metric_name values:\n"
        "- personal_loan_percent (for personal loans as % of total advances)\n"
        "- credit_card_npa (for credit card NPA % or retail/unsecured NPA proxy)\n"
        "- unsecured_loan_percent (for total unsecured loans as % of advances)\n"
        "- personal_loan_growth (for YoY personal loan growth rate)\n"
    ),
    "market_risk": (
        "For market_risk, use EXACTLY these metric_name values:\n"
        "- var_10day (for Value at Risk)\n"
        "- duration_gap (for duration gap in years)\n"
        "- trading_book_size (for trading book / HFT portfolio size)\n"
        "- mtm_losses (for mark-to-market losses)\n"
    ),
    "operational_risk": (
        "For operational_risk, use EXACTLY these metric_name values:\n"
        "- oprisk_rwa (for operational risk RWA)\n"
        "- fraud_losses (for fraud-related losses)\n"
        "- cyber_incidents (for number of cyber incidents)\n"
        "- bcp_status (for business continuity plan status)\n"
    ),
    "asset_quality_trend": (
        "For asset_quality_trend, use EXACTLY these metric_name values:\n"
        "- npa_opening (for opening NPA balance)\n"
        "- npa_additions (for fresh NPA additions / slippages)\n"
        "- npa_recoveries (for recoveries and upgradations)\n"
        "- npa_closing (for closing NPA balance)\n"
        "- writeoff_rate (for write-off rate or amount)\n"
    ),
}


def get_metric_names_instruction(theme: str) -> str:
    """Get the metric naming instruction for a given theme."""
    return METRIC_NAMES_BY_THEME.get(
        theme,
        "Use short snake_case metric names (e.g. gross_npa_ratio, not long descriptive names)."
    )


BANK_CONTEXT_TEMPLATE = """
[BEGIN {bank_name} CONTEXT — {fiscal_year} — {document_type}]
{chunks}
[END {bank_name} CONTEXT]
"""

CHUNK_TEMPLATE = """[Page {page_number}] [Section: {section_header}]
{content}
"""

PEER_SYNTHESIS_PROMPT = """You are a senior financial analyst. Given the individual thematic
extractions below for multiple banks, synthesize a cross-cutting comparative analysis.

RULES:
1. Compare the banks on the specific metrics extracted.
2. Highlight divergences and convergences.
3. Note any banks with missing disclosures.
4. Keep the synthesis under 2000 characters.
5. Ground every claim in the extracted data — do not add external knowledge.

THEME: {theme}
PEER GROUP: {peer_group}

INDIVIDUAL EXTRACTIONS:
{extractions_json}

Write a concise cross-cutting analysis:
"""
