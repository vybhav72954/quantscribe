import streamlit as st
st.set_page_config(page_title="QuantScribe | Thematic Peer Analysis", layout="wide")

import sys, os
from pathlib import Path

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Load secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY! Please add it to Streamlit Cloud Secrets.")
    st.stop()

import pandas as pd
import plotly.express as px

from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever
from quantscribe.llm.extraction_chain import build_extraction_chain
from quantscribe.llm.peer_comparison import run_peer_comparison


@st.cache_resource
def load_pipeline():
    embedder = EmbeddingPipeline()
    extraction_chain = build_extraction_chain(max_retries=3)
    return embedder, extraction_chain


def get_available_indices(index_dir="indices/active"):
    path = Path(index_dir)
    if not path.exists():
        return []
    banks = set()
    for f in path.glob("*.faiss"):
        # Works for any doc type: strip everything from _annual_report, _earnings_call, etc.
        stem = f.stem
        for suffix in ["_annual_report", "_earnings_call", "_investor_presentation"]:
            if suffix in stem:
                banks.add(stem.split(suffix)[0])
                break
    return sorted(banks)


# ── Sidebar ──
st.sidebar.title("🔍 QuantScribe Controls")
theme_options = [
    "credit_risk", "liquidity_risk", "unsecured_lending",
    "capital_adequacy", "market_risk", "asset_quality_trend",
]
selected_theme = st.sidebar.selectbox("Select Macro Theme", theme_options)

available_banks = get_available_indices()
selected_banks = st.sidebar.multiselect(
    "Select Banks for Peer Analysis",
    available_banks,
    default=available_banks[:2] if len(available_banks) >= 2 else available_banks,
)
top_k = st.sidebar.slider("Chunks per Bank (Top-K)", 3, 10, 5)

st.sidebar.divider()
if st.sidebar.button("Clear Cache"):
    st.cache_resource.clear()
    st.rerun()

# ── Main UI ──
st.title("📊 Automated Thematic Peer Analysis")
st.markdown(f"**Theme:** `{selected_theme}` | **Peer Group:** {', '.join(selected_banks) if selected_banks else '_None selected_'}")

if not selected_banks:
    st.warning("Please select at least one bank from the sidebar.")
    st.stop()

if len(selected_banks) < 2:
    st.warning("`PeerComparisonReport` requires at least 2 banks (min_length=2 on peer_group).")
    st.stop()

embedder, extraction_chain = load_pipeline()

if st.button("Generate Comparison Report", type="primary"):
    with st.spinner(f"Extracting `{selected_theme}` data from annual reports..."):
        try:
            # Load indices — key MUST be index_name, not bank_name
            bank_indices = {}
            for bank in selected_banks:
                index_name = f"{bank}_annual_report_FY25"
                idx = BankIndex(index_name)
                idx.load("indices/active")
                bank_indices[index_name] = idx

            retriever = PeerGroupRetriever(bank_indices)

            report = run_peer_comparison(
                theme=selected_theme,
                peer_group=selected_banks,
                retriever=retriever,
                embedding_pipeline=embedder,
                extraction_chain=extraction_chain,
                top_k_per_bank=top_k,
            )

            # ── Risk Score Chart ──
            st.subheader("Risk Score Comparison")
            ranking_df = pd.DataFrame([r.model_dump() for r in report.peer_ranking])
            fig = px.bar(
                ranking_df, x="bank", y="risk_score",
                color="risk_score", color_continuous_scale="RdYlGn_r",
                range_color=[0, 10],
                title="Risk Score by Bank (Lower = Safer)",
                labels={"risk_score": "Risk Score (0–10)", "bank": "Bank"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Per-Bank Tabs ──
            st.subheader("Detailed Bank Analysis")
            tabs = st.tabs([ext.bank_name for ext in report.extractions])

            for i, ext in enumerate(report.extractions):
                with tabs[i]:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Risk Score", f"{ext.risk_score:.1f}/10", ext.risk_rating)
                        st.metric("Sentiment", f"{ext.sentiment_score:.2f}")
                    with col2:
                        st.write("**Executive Summary:**")
                        st.info(ext.summary)

                    st.write("**Extracted Metrics & Citations:**")
                    metrics_data = []
                    for m in ext.extracted_metrics:
                        val = m.metric_value if m.metric_value is not None else m.qualitative_value
                        metrics_data.append({
                            "Metric": m.metric_name,
                            "Value": f"{val} {m.metric_unit or ''}".strip(),
                            "Confidence": m.confidence,
                            "Page": m.citation.page_number,
                            "Source Text": m.citation.source_excerpt,
                        })

                    metrics_df = pd.DataFrame(metrics_data)
                    st.table(metrics_df.drop(columns=["Source Text"]))

                    with st.expander("View Full Citations"):
                        for m in metrics_data:
                            st.markdown(f"**{m['Metric']}** *(Page {m['Page']})*: _{m['Source Text']}_")

            # ── Cross-Cutting Insights ──
            if report.cross_cutting_insights:
                st.divider()
                st.subheader("🎯 Cross-Cutting Insights")
                st.success(report.cross_cutting_insights)

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            st.exception(e)
