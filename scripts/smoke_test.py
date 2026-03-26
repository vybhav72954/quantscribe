from quantscribe.etl.pipeline import run_etl_pipeline

chunks = run_etl_pipeline(
    pdf_path="data/pdfs/HDFC Bank Report.pdf",
    bank_name="HDFC_BANK",
    document_type="annual_report",
    fiscal_year="FY25",
    page_range=(49, 60),
)
print(f"Chunks: {len(chunks)}")
for c in chunks[:5]:
    print(f"  P{c.metadata.page_number} [{c.content_type}] section={c.metadata.section_header} words={c.metadata.token_count}")
    