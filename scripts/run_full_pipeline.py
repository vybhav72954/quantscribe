"""
Full end-to-end QuantScribe pipeline.

Loads FAISS indices, runs peer comparison, evaluates against gold standard.
Run AFTER Team A has produced FAISS indices AND Team B has implemented extraction chain.

Usage:
    python scripts/run_full_pipeline.py --theme credit_risk
    python scripts/run_full_pipeline.py --theme capital_adequacy --banks HDFC_BANK SBI ICICI_BANK
    python scripts/run_full_pipeline.py --theme credit_risk --skip-eval
"""

import argparse
import json
import glob
import os
import sys
import time

sys.path.insert(0, ".")

from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever
from quantscribe.llm.extraction_chain import build_extraction_chain
from quantscribe.llm.peer_comparison import run_peer_comparison
from quantscribe.schemas.evaluation import EvalTestCase
from quantscribe.schemas.extraction import ThematicExtraction
from quantscribe.evaluation.numerical_eval import evaluate_numerical_accuracy


def main():
    parser = argparse.ArgumentParser(description="Run QuantScribe full pipeline")
    parser.add_argument("--theme", required=True, help="Macro theme (e.g., credit_risk, capital_adequacy)")
    parser.add_argument("--banks", nargs="+", default=["HDFC_BANK", "SBI"], help="Banks to compare")
    parser.add_argument("--index-dir", default="indices/active", help="Directory containing FAISS indices")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K chunks per bank")
    parser.add_argument("--retries", type=int, default=3, help="LLM retry attempts")
    parser.add_argument("--skip-eval", action="store_true", help="Skip gold-standard evaluation")
    args = parser.parse_args()

    # ══════════════════════════════════════════════════
    # PHASE 1: Load Models & Indices
    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Loading models and indices")
    print("=" * 60)

    print("Loading embedding model...")
    embedder = EmbeddingPipeline()

    print("Loading FAISS indices...")
    bank_indices = {}
    for bank in args.banks:
        index_name = f"{bank}_annual_report_FY25"
        index = BankIndex(index_name)
        try:
            index.load(args.index_dir)
            bank_indices[bank] = index
            print(f"  {bank}: {index.size} vectors loaded")
        except FileNotFoundError:
            print(f"  WARNING: No index found for {bank}, skipping")

    if len(bank_indices) < 2:
        print("ERROR: Need at least 2 banks with loaded indices for peer comparison")
        sys.exit(1)

    retriever = PeerGroupRetriever(bank_indices)
    available_banks = list(bank_indices.keys())

    print(f"\nReady: {len(available_banks)} banks loaded")

    # ══════════════════════════════════════════════════
    # PHASE 2: Run Peer Comparison
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 2: Running {args.theme} peer comparison")
    print("=" * 60)

    print("Building extraction chain...")
    extraction_chain = build_extraction_chain(max_retries=args.retries)

    start_time = time.time()
    report = run_peer_comparison(
        theme=args.theme,
        peer_group=available_banks,
        retriever=retriever,
        embedding_pipeline=embedder,
        extraction_chain=extraction_chain,
        top_k_per_bank=args.top_k,
    )
    elapsed = time.time() - start_time

    # ── Print Results ──
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Theme: {report.query_theme}")
    print(f"Peer Group: {report.peer_group}")

    print(f"\nRankings:")
    for r in report.peer_ranking:
        print(f"  #{r.rank} {r.bank}: risk_score={r.risk_score}")

    for ext in report.extractions:
        print(f"\n--- {ext.bank_name} ---")
        print(f"  Risk Score: {ext.risk_score} ({ext.risk_rating})")
        print(f"  Sentiment: {ext.sentiment_score:.2f}")
        print(f"  Summary: {ext.summary[:150]}...")
        print(f"  Metrics:")
        for m in ext.extracted_metrics:
            val = m.metric_value if m.metric_value is not None else m.qualitative_value
            print(f"    {m.metric_name}: {val} {m.metric_unit or ''} (confidence: {m.confidence})")
            print(f"      Page {m.citation.page_number}, score {m.citation.relevance_score:.2f}")

    print(f"\nCross-cutting insights:\n  {report.cross_cutting_insights[:300]}...")

    # ── Save Report ──
    os.makedirs("data/reports", exist_ok=True)
    report_path = f"data/reports/{args.theme}_peer_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"\nReport saved to {report_path}")

    # ══════════════════════════════════════════════════
    # PHASE 3: Evaluate Against Gold Standard
    # ══════════════════════════════════════════════════
    if args.skip_eval:
        print("\nSkipping evaluation (--skip-eval flag set)")
        return

    print("\n" + "=" * 60)
    print("PHASE 3: Gold-standard evaluation")
    print("=" * 60)

    # Load gold-standard test cases
    test_files = glob.glob("eval/gold_standard/*.json")
    if not test_files:
        print("No gold-standard test cases found in eval/gold_standard/")
        print("Create them manually and re-run without --skip-eval")
        return

    test_cases: list[EvalTestCase] = []
    for f in test_files:
        with open(f) as fh:
            test_cases.append(EvalTestCase(**json.load(fh)))
    print(f"Loaded {len(test_cases)} gold-standard test cases")

    # Match extractions to test cases and evaluate
    total_metrics = 0
    correct_metrics = 0
    results_by_bank: dict[str, dict] = {}

    for ext in report.extractions:
        matching_tests = [
            tc for tc in test_cases
            if tc.bank_name == ext.bank_name and tc.query_theme == ext.theme
        ]

        if not matching_tests:
            print(f"\n  No gold standard for {ext.bank_name}/{ext.theme}, skipping")
            continue

        for gold in matching_tests:
            print(f"\n  Evaluating: {gold.test_id}")
            results = evaluate_numerical_accuracy(ext, gold)

            bank_results: dict[str, str] = {}
            for metric, is_correct in results.items():
                total_metrics += 1
                if is_correct:
                    correct_metrics += 1
                status = "PASS" if is_correct else "FAIL"
                expected = gold.expected_metrics[metric]
                actual = next(
                    (m.metric_value for m in ext.extracted_metrics if m.metric_name == metric),
                    None,
                )
                print(f"    [{status}] {metric}: expected={expected}, got={actual}")
                bank_results[metric] = status

            results_by_bank[gold.test_id] = bank_results

    # ── Final Score ──
    print("\n" + "=" * 60)
    print("FINAL SCORE")
    print("=" * 60)

    if total_metrics > 0:
        accuracy = 100 * correct_metrics / total_metrics
        print(f"Numerical Accuracy: {correct_metrics}/{total_metrics} ({accuracy:.0f}%)")
        print(f"Target: >= 90%")
        print(f"Status: {'PASS' if accuracy >= 90 else 'FAIL'}")
    else:
        print("No metrics evaluated — check that gold-standard themes match the query theme")

    # Schema compliance
    schema_valid = all(isinstance(ext, ThematicExtraction) for ext in report.extractions)
    print(f"Schema Compliance: {'PASS' if schema_valid else 'FAIL'} (100% required)")


if __name__ == "__main__":
    main()
    