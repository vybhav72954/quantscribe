"""
Run ETL pipeline on a bank's annual report.

Usage:
    python scripts/run_etl.py --bank HDFC_BANK --pdf "data/pdfs/HDFC Bank Report.pdf" --fy FY25
    python scripts/run_etl.py --bank SBI --pdf "data/pdfs/SBI Bank Report.pdf" --fy FY25

    # Optional: process only a range of pages (for testing)
    python scripts/run_etl.py --bank HDFC_BANK --pdf "data/pdfs/HDFC Bank Report.pdf" --fy FY25 --pages 49-60
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from quantscribe.etl.pipeline import run_etl_pipeline, save_chunks_to_json
from quantscribe.schemas.etl import TextChunk


def main():
    parser = argparse.ArgumentParser(description="Run QuantScribe ETL pipeline on a PDF")
    parser.add_argument("--bank", required=True, help="Bank name (e.g., HDFC_BANK, SBI)")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--fy", default="FY25", help="Fiscal year (default: FY25)")
    parser.add_argument("--doc-type", default="annual_report", help="Document type (default: annual_report)")
    parser.add_argument("--pages", default=None, help="Page range e.g. 49-60 (default: all pages)")
    args = parser.parse_args()

    # Parse page range if provided
    page_range = None
    if args.pages:
        start, end = args.pages.split("-")
        page_range = (int(start), int(end))
        print(f"Processing pages {start}-{end} only")

    # Run pipeline
    print(f"Processing {args.bank} — {args.pdf}")
    start_time = time.time()

    chunks = run_etl_pipeline(
        pdf_path=args.pdf,
        bank_name=args.bank,
        document_type=args.doc_type,
        fiscal_year=args.fy,
        page_range=page_range,
    )

    elapsed = time.time() - start_time
    print(f"\nDone: {len(chunks)} chunks in {elapsed / 60:.1f} minutes")

    # Save chunks
    output_path = f"data/chunks/{args.bank}_{args.doc_type}_{args.fy}_chunks.json"
    save_chunks_to_json(chunks, output_path)
    print(f"Saved to {output_path}")

    # Quality check
    print("\n=== Quality Check ===")
    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.content_type] = type_counts.get(c.content_type, 0) + 1
    print(f"Chunk types: {type_counts}")

    sections = sorted(set(c.metadata.section_header for c in chunks if c.metadata.section_header))
    print(f"Sections found ({len(sections)}): {sections[:10]}{'...' if len(sections) > 10 else ''}")

    pages = sorted(set(c.metadata.page_number for c in chunks))
    print(f"Pages covered: {len(pages)} pages")

    valid = all(isinstance(c, TextChunk) for c in chunks)
    print(f"Pydantic valid: {valid} ({len(chunks)}/{len(chunks)})")


if __name__ == "__main__":
    main()
    