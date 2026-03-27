"""
QuantScribe Full Evaluation Suite.

Runs all three evaluation layers:
1. Numerical Accuracy — exact metric value matching (±0.5% tolerance)
2. RAGAS — context precision + faithfulness (LLM-as-judge)
3. DeepEval — faithfulness + answer relevancy (LLM-as-judge)

Prerequisites:
- FAISS indices must be in indices/active/
- A pipeline run must have completed (report JSON in data/reports/)
  OR pass --live to run extraction live

Usage:
    # Evaluate a saved report
    python scripts/run_eval.py --theme credit_risk

    # Run live extraction + evaluation
    python scripts/run_eval.py --theme credit_risk --live

    # All themes
    python scripts/run_eval.py --theme credit_risk capital_adequacy liquidity_risk unsecured_lending --live

    # Skip RAGAS/DeepEval (just numerical)
    python scripts/run_eval.py --theme credit_risk --numerical-only
"""

import argparse
import json
import glob
import os
import sys
import time

sys.path.insert(0, ".")

from quantscribe.config import get_settings, THEME_TAXONOMY
from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever
from quantscribe.llm.extraction_chain import build_extraction_chain
from quantscribe.llm.peer_comparison import run_peer_comparison, _build_query_text, _format_bank_context
from quantscribe.schemas.evaluation import EvalTestCase
from quantscribe.schemas.extraction import ThematicExtraction
from quantscribe.schemas.report import PeerComparisonReport
from quantscribe.evaluation.numerical_eval import evaluate_numerical_accuracy


def main():
    parser = argparse.ArgumentParser(description="QuantScribe Full Evaluation Suite")
    parser.add_argument(
        "--theme", nargs="+", required=True,
        help="Theme(s) to evaluate (e.g., credit_risk capital_adequacy)",
    )
    parser.add_argument("--banks", nargs="+", default=["HDFC_BANK", "SBI"])
    parser.add_argument("--index-dir", default="indices/active")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="Run live extraction (else load saved report)")
    parser.add_argument("--numerical-only", action="store_true", help="Skip RAGAS/DeepEval")
    args = parser.parse_args()

    # ═══════════════════════════════════════════════
    # SETUP
    # ═══════════════════════════════════════════════
    print("=" * 70)
    print("  QuantScribe — Full Evaluation Suite")
    print("=" * 70)

    print("\nLoading embedding model...")
    embedder = EmbeddingPipeline()

    print("Loading FAISS indices...")
    bank_indices = {}
    for bank in args.banks:
        index_name = f"{bank}_annual_report_FY25"
        index = BankIndex(index_name)
        try:
            index.load(args.index_dir)
            bank_indices[bank] = index
            print(f"  {bank}: {index.size} vectors")
        except FileNotFoundError:
            print(f"  WARNING: {bank} index not found, skipping")

    if len(bank_indices) < 2:
        print("ERROR: Need at least 2 bank indices")
        sys.exit(1)

    retriever = PeerGroupRetriever(bank_indices)
    available_banks = list(bank_indices.keys())

    # Load gold standard
    test_files = glob.glob("eval/gold_standard/*.json")
    all_gold: list[EvalTestCase] = []
    for f in test_files:
        with open(f) as fh:
            all_gold.append(EvalTestCase(**json.load(fh)))
    print(f"Gold standard: {len(all_gold)} test cases loaded")

    # ═══════════════════════════════════════════════
    # RUN PER THEME
    # ═══════════════════════════════════════════════
    all_scores: list[dict] = []

    for theme in args.theme:
        print(f"\n{'=' * 70}")
        print(f"  THEME: {theme}")
        print(f"{'=' * 70}")

        # ── Get report (live or saved) ──
        if args.live:
            print("\n  Running live extraction...")
            chain = build_extraction_chain(max_retries=3)
            report = run_peer_comparison(
                theme=theme,
                peer_group=available_banks,
                retriever=retriever,
                embedding_pipeline=embedder,
                extraction_chain=chain,
                top_k_per_bank=args.top_k,
            )
        else:
            report_path = f"data/reports/{theme}_peer_comparison.json"
            if not os.path.exists(report_path):
                print(f"  Report not found at {report_path}")
                print(f"  Run the pipeline first or use --live")
                continue
            with open(report_path) as f:
                report = PeerComparisonReport(**json.load(f))
            print(f"  Loaded saved report from {report_path}")

        # ── Retrieve contexts for eval ──
        query_text = _build_query_text(theme)
        query_vector = embedder.embed_query(query_text)
        all_results = retriever.retrieve(query_vector, available_banks, top_k_per_bank=args.top_k)

        # ── Evaluate each bank ──
        for ext in report.extractions:
            bank = ext.bank_name
            print(f"\n  --- {bank} ---")

            # Get gold standard
            matching_gold = [
                g for g in all_gold
                if g.bank_name == bank and g.query_theme == theme
            ]

            # Build context and response strings for RAGAS/DeepEval
            bank_results = all_results.get(bank, [])
            retrieved_contexts = [
                r["metadata"].get("content", "") for r in bank_results
                if r["metadata"].get("content")
            ]
            llm_response = _extraction_to_text(ext)

            # Track scores
            theme_scores: dict = {
                "theme": theme,
                "bank": bank,
            }

            # ── Layer 1: Numerical Accuracy ──
            if matching_gold:
                gold = matching_gold[0]
                num_results = evaluate_numerical_accuracy(ext, gold)
                passed = sum(num_results.values())
                total = len(num_results)
                accuracy = passed / max(total, 1)
                theme_scores["numerical_accuracy"] = round(accuracy, 4)

                print(f"  Numerical Accuracy: {passed}/{total} ({accuracy:.0%})")
                for metric, ok in num_results.items():
                    expected = gold.expected_metrics[metric]
                    actual = next(
                        (m.metric_value for m in ext.extracted_metrics if m.metric_name == metric),
                        None,
                    )
                    status = "PASS" if ok else "FAIL"
                    print(f"    [{status}] {metric}: expected={expected}, got={actual}")

                # Retrieval hit rate
                retrieved_pages = {r["metadata"]["page_number"] for r in bank_results}
                expected_pages = set(gold.expected_pages)
                hit = bool(retrieved_pages & expected_pages)
                theme_scores["retrieval_hit"] = hit
                print(f"  Retrieval Hit: {'YES' if hit else 'NO'} "
                      f"(retrieved pages: {sorted(retrieved_pages)}, "
                      f"expected: {sorted(expected_pages)})")
            else:
                print(f"  No gold standard for {bank}/{theme}")
                theme_scores["numerical_accuracy"] = -1.0

            # ── Layer 2: RAGAS ──
            if not args.numerical_only and retrieved_contexts:
                print(f"  Running RAGAS evaluation...")
                try:
                    from quantscribe.evaluation.ragas_eval import run_ragas_evaluation
                    ragas_scores = run_ragas_evaluation(
                        theme=theme,
                        bank_name=bank,
                        query=query_text,
                        retrieved_contexts=retrieved_contexts,
                        llm_response=llm_response,
                    )
                    theme_scores.update({f"ragas_{k}": v for k, v in ragas_scores.items()})
                    for k, v in ragas_scores.items():
                        score_str = f"{v:.4f}" if v >= 0 else "FAILED"
                        print(f"    RAGAS {k}: {score_str}")
                except Exception as e:
                    print(f"    RAGAS failed: {str(e)[:100]}")
                    theme_scores["ragas_context_precision"] = -1.0
                    theme_scores["ragas_faithfulness"] = -1.0

            # ── Layer 3: DeepEval ──
            if not args.numerical_only and retrieved_contexts:
                print(f"  Running DeepEval evaluation...")
                try:
                    from quantscribe.evaluation.deepeval_eval import run_deepeval_evaluation
                    deep_scores = run_deepeval_evaluation(
                        theme=theme,
                        bank_name=bank,
                        query=query_text,
                        retrieved_contexts=retrieved_contexts,
                        llm_response=llm_response,
                    )
                    theme_scores.update({f"deepeval_{k}": v for k, v in deep_scores.items()})
                    for k, v in deep_scores.items():
                        score_str = f"{v:.4f}" if v >= 0 else "FAILED"
                        print(f"    DeepEval {k}: {score_str}")
                except Exception as e:
                    print(f"    DeepEval failed: {str(e)[:100]}")
                    theme_scores["deepeval_faithfulness"] = -1.0
                    theme_scores["deepeval_answer_relevancy"] = -1.0

            all_scores.append(theme_scores)

    # ═══════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  FINAL SCORECARD")
    print(f"{'=' * 70}")

    # Aggregate
    num_accuracies = [s["numerical_accuracy"] for s in all_scores if s.get("numerical_accuracy", -1) >= 0]
    avg_num = sum(num_accuracies) / max(len(num_accuracies), 1) if num_accuracies else 0

    print(f"\n  Numerical Accuracy (avg): {avg_num:.0%}")
    print(f"    Target: >= 90%")
    print(f"    Status: {'PASS' if avg_num >= 0.9 else 'FAIL'}")

    retrieval_hits = [s.get("retrieval_hit", False) for s in all_scores if "retrieval_hit" in s]
    if retrieval_hits:
        hit_rate = sum(retrieval_hits) / len(retrieval_hits)
        print(f"\n  Retrieval Hit Rate: {hit_rate:.0%}")
        print(f"    Target: >= 75%")
        print(f"    Status: {'PASS' if hit_rate >= 0.75 else 'FAIL'}")

    if not args.numerical_only:
        # RAGAS
        ragas_cp = [s["ragas_context_precision"] for s in all_scores if s.get("ragas_context_precision", -1) >= 0]
        ragas_f = [s["ragas_faithfulness"] for s in all_scores if s.get("ragas_faithfulness", -1) >= 0]

        if ragas_cp:
            avg_cp = sum(ragas_cp) / len(ragas_cp)
            print(f"\n  RAGAS Context Precision (avg): {avg_cp:.4f}")
            print(f"    Target: >= 0.70")
            print(f"    Status: {'PASS' if avg_cp >= 0.70 else 'FAIL'}")

        if ragas_f:
            avg_f = sum(ragas_f) / len(ragas_f)
            print(f"\n  RAGAS Faithfulness (avg): {avg_f:.4f}")
            print(f"    Target: >= 0.85")
            print(f"    Status: {'PASS' if avg_f >= 0.85 else 'FAIL'}")

        # DeepEval
        deep_f = [s["deepeval_faithfulness"] for s in all_scores if s.get("deepeval_faithfulness", -1) >= 0]
        deep_r = [s["deepeval_answer_relevancy"] for s in all_scores if s.get("deepeval_answer_relevancy", -1) >= 0]

        if deep_f:
            avg_df = sum(deep_f) / len(deep_f)
            print(f"\n  DeepEval Faithfulness (avg): {avg_df:.4f}")
            print(f"    Target: >= 0.85")
            print(f"    Status: {'PASS' if avg_df >= 0.85 else 'FAIL'}")

        if deep_r:
            avg_dr = sum(deep_r) / len(deep_r)
            print(f"\n  DeepEval Answer Relevancy (avg): {avg_dr:.4f}")
            print(f"    Target: >= 0.80")
            print(f"    Status: {'PASS' if avg_dr >= 0.80 else 'FAIL'}")

    # Save results
    os.makedirs("data/eval_results", exist_ok=True)
    results_path = f"data/eval_results/eval_{'_'.join(args.theme)}.json"
    with open(results_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\n  Results saved to {results_path}")


def _extraction_to_text(ext: ThematicExtraction) -> str:
    """Convert a ThematicExtraction to a text string for RAGAS/DeepEval."""
    lines = [
        f"Bank: {ext.bank_name}",
        f"Theme: {ext.theme}",
        f"Risk Score: {ext.risk_score} ({ext.risk_rating})",
        f"Summary: {ext.summary}",
        "Extracted Metrics:",
    ]
    for m in ext.extracted_metrics:
        val = m.metric_value if m.metric_value is not None else m.qualitative_value
        lines.append(f"  {m.metric_name}: {val} {m.metric_unit or ''}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()