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

QUERIED THEME: {theme}

{bank_contexts}
"""

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