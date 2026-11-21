"""Chunking Analysis page for FraudResearchAssistant.

Uses only Notebook 09 95-question retrieval exports. Metrics are computed on
70 answerable questions; latency rows were collected for 95 questions.
No API key, LLM inference, or document-index download is required.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from core.ui import apply_theme, page_header

apply_theme()

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"
REQUIRED = [
    "article_page_metrics_summary.csv",
    "article_page_metrics_per_question.csv",
    "retrieval_timings.csv",
    "retrieval_failure_analysis.csv",
]
STRATEGIES = ["fixed_size", "sentence_based", "recursive", "section_aware"]
METHODS = ["bm25", "dense_faiss", "hybrid_rrf", "hybrid_reranked"]
STRATEGY_NAMES = {
    "fixed_size": "Fixed-size",
    "sentence_based": "Sentence-based",
    "recursive": "Recursive",
    "section_aware": "Section-aware",
}
METHOD_NAMES = {
    "bm25": "BM25",
    "dense_faiss": "Dense FAISS",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_reranked": "Hybrid + reranker",
}
METRICS = {
    "Article Hit@K": "article_hit_at_{k}_passages",
    "Article Recall@K": "article_recall_at_{k}",
    "Article Precision@K": "article_precision_at_{k}",
    "Article MRR@K": "article_mrr_at_{k}",
    "Article nDCG@K": "article_ndcg_at_{k}",
    "Page-overlap Hit@K": "page_overlap_hit_at_{k}",
}


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, timestamp_ns: int) -> pd.DataFrame:
    """Include file modification time in key to refresh after file replacement."""
    _ = timestamp_ns
    return pd.read_csv(path)


def read_report(filename: str) -> pd.DataFrame:
    path = REPORTS / filename
    return read_csv_cached(str(path), path.stat().st_mtime_ns)


def display_name(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, str(value))


def csv_download(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        mime="text/csv",
        file_name=filename,
    )


page_header(
    "Chunking Analysis",
    "Compare verified chunking strategies, retrieval quality, and recorded latency from Notebook 09.",
    "EXPERIMENTAL DASHBOARD",
)
st.info(
    "**Metric scope:** Retrieval quality is computed from the **70 answerable** "
    "questions with gold evidence. The retrieval-timing table covers "
    "**all 95 questions**. This dashboard reports *retrieval* quality, "
    "not final LLM answer quality."
)

missing = [filename for filename in REQUIRED if not (REPORTS / filename).is_file()]
if missing:
    st.error("Required Notebook 09 CSV files are missing from `data/reports/`.")
    st.code("\n".join(missing), language="text")
    st.stop()

try:
    summary = read_report(REQUIRED[0])
    per_question = read_report(REQUIRED[1])
    timings = read_report(REQUIRED[2])
    failures = read_report(REQUIRED[3])
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not read the CSV exports: {exc}")
    st.stop()

needed = {
    "article_page_metrics_summary.csv": (
        summary, {"chunking_strategy", "method"}
        | {template.format(k=k) for template in METRICS.values() for k in (1, 3, 5, 10)}
    ),
    "article_page_metrics_per_question.csv": (
        per_question, {"question_id", "chunking_strategy", "method", "article_ndcg_at_5"}
    ),
    "retrieval_timings.csv": (
        timings, {"question_id", "chunking_strategy", "method", "retrieval_latency_ms"}
    ),
    "retrieval_failure_analysis.csv": (
        failures, {"question_id", "chunking_strategy", "method"}
    ),
}
for filename, (frame, fields) in needed.items():
    absent = sorted(fields - set(frame.columns))
    if absent:
        st.error(f"`{filename}` is missing expected columns: {', '.join(absent)}")
        st.stop()

if summary.duplicated(["chunking_strategy", "method"]).any():
    st.error("Duplicate strategy–method pairs found in the summary CSV.")
    st.stop()

quality_question_count = per_question["question_id"].nunique()
timing_question_count = timings["question_id"].nunique()
q1, q2, q3, q4 = st.columns(4)
q1.metric("Chunking strategies", summary["chunking_strategy"].nunique())
q2.metric("Retrieval methods", summary["method"].nunique())
q3.metric("Quality-evaluated questions", quality_question_count)
q4.metric("Timing-evaluated questions", timing_question_count)

st.divider()
controls_a, controls_b, controls_c = st.columns([1.2, 1.2, 1.7])
with controls_a:
    k = st.selectbox("Top-K passages", (1, 3, 5, 10), index=2)
with controls_b:
    metric_label = st.selectbox("Retrieval metric", list(METRICS.keys()), index=4)
with controls_c:
    method = st.selectbox(
        "Focus retrieval method", METHODS,
        format_func=lambda key: display_name(key, METHOD_NAMES),
        index=2,
    )
metric_col = METRICS[metric_label].format(k=k)

summary_plot = summary.copy()
summary_plot[metric_col] = pd.to_numeric(summary_plot[metric_col], errors="coerce")
summary_plot["Strategy"] = summary_plot["chunking_strategy"].map(STRATEGY_NAMES)
summary_plot["Method"] = summary_plot["method"].map(METHOD_NAMES)

st.subheader(f"Chunking performance for {METHOD_NAMES[method]}")
selected = summary_plot[summary_plot["method"].eq(method)].copy()
selected = selected.set_index("chunking_strategy").reindex(STRATEGIES).reset_index()
selected["Strategy"] = selected["chunking_strategy"].map(STRATEGY_NAMES)

fig = px.bar(
    selected, x="Strategy", y=metric_col, text=metric_col,
    labels={metric_col: f"{metric_label} (0–1)"},
    category_orders={"Strategy": [STRATEGY_NAMES[s] for s in STRATEGIES]},
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(yaxis_range=[0, 1.09], margin=dict(t=15, b=15))
st.plotly_chart(fig, use_container_width=True)

valid = selected.dropna(subset=[metric_col])
if not valid.empty:
    max_val = valid[metric_col].max()
    leaders = valid.loc[valid[metric_col].eq(max_val), "Strategy"].tolist()
    st.caption(
        f"Highest observed {metric_label} for {METHOD_NAMES[method]} "
        f"at K={k}: {', '.join(leaders)} ({max_val:.3f}). "
        "This is a descriptive ranking, not a significance test."
    )

st.subheader("All methods × chunking strategies")
compare = summary_plot[["Strategy", "Method", metric_col]].copy()
fig_all = px.bar(
    compare,
    x="Strategy", y=metric_col, color="Method", barmode="group",
    category_orders={
        "Strategy": [STRATEGY_NAMES[s] for s in STRATEGIES],
        "Method": [METHOD_NAMES[m] for m in METHODS],
    },
    labels={metric_col: f"{metric_label} (0–1)"},
)
fig_all.update_layout(yaxis_range=[0, 1.05], margin=dict(t=15, b=15))
st.plotly_chart(fig_all, use_container_width=True)

st.subheader("Retrieval performance table")
table_cols = [
    "chunking_strategy", "method", metric_col,
    f"article_hit_at_{k}_passages", f"page_overlap_hit_at_{k}",
    f"article_recall_at_{k}", f"article_precision_at_{k}",
]
# Deduplicate when the selected metric is also a standard table metric.
table_cols = list(dict.fromkeys(table_cols))
full_table = summary[table_cols].copy()
full_table["chunking_strategy"] = full_table["chunking_strategy"].map(STRATEGY_NAMES)
full_table["method"] = full_table["method"].map(METHOD_NAMES)
full_table = full_table.rename(columns={
    "chunking_strategy": "Chunking strategy", "method": "Retrieval method"
})
st.dataframe(full_table, hide_index=True, use_container_width=True)
csv_download(full_table, "Download comparison table", f"chunking_retrieval_metrics_at_{k}.csv")

st.divider()
st.subheader("Retrieval latency by chunking strategy")
st.caption(
    "These values are taken from Notebook 09's recorded retrieval latency. "
    "Do not add query-encoding time again unless its timing definition confirms "
    "that it was measured separately."
)
latency = timings.copy()
latency["retrieval_latency_ms"] = pd.to_numeric(
    latency["retrieval_latency_ms"], errors="coerce"
)
latency = latency.dropna(subset=["retrieval_latency_ms"])
latency_summary = latency.groupby(["chunking_strategy", "method"], as_index=False).agg(
    n_timed_questions=("question_id", "nunique"),
    mean_latency_ms=("retrieval_latency_ms", "mean"),
    median_latency_ms=("retrieval_latency_ms", "median"),
    p95_latency_ms=("retrieval_latency_ms", lambda s: s.quantile(0.95)),
)
latency_summary["Strategy"] = latency_summary["chunking_strategy"].map(STRATEGY_NAMES)
latency_summary["Method"] = latency_summary["method"].map(METHOD_NAMES)
fig_latency = px.bar(
    latency_summary, x="Strategy", y="mean_latency_ms", color="Method", barmode="group",
    category_orders={"Strategy": [STRATEGY_NAMES[s] for s in STRATEGIES]},
    labels={"mean_latency_ms": "Mean retrieval time (ms)"},
)
fig_latency.update_layout(margin=dict(t=15, b=15))
st.plotly_chart(fig_latency, use_container_width=True)
st.dataframe(
    latency_summary[["Strategy", "Method", "n_timed_questions", "mean_latency_ms",
                     "median_latency_ms", "p95_latency_ms"]].rename(columns={
        "n_timed_questions": "Questions", "mean_latency_ms": "Mean (ms)",
        "median_latency_ms": "Median (ms)", "p95_latency_ms": "P95 (ms)"
    }),
    hide_index=True, use_container_width=True,
)

st.divider()
st.subheader("Inspect individual retrieval questions")
question_options = sorted(per_question["question_id"].astype(str).unique())
selected_qid = st.selectbox("Question ID (answerable questions only)", question_options)
case_rows = per_question.loc[
    per_question["question_id"].astype(str).eq(selected_qid),
    ["chunking_strategy", "method", "article_ndcg_at_5", "article_hit_at_5_passages",
     "page_overlap_hit_at_5", "article_recall_at_5"],
].copy()
case_rows["chunking_strategy"] = case_rows["chunking_strategy"].map(STRATEGY_NAMES)
case_rows["method"] = case_rows["method"].map(METHOD_NAMES)
st.dataframe(case_rows, hide_index=True, use_container_width=True)

with st.expander("Notebook 09 retrieval failure examples"):
    st.caption(
        "This is the specific six-row failure-analysis export produced for "
        "section-aware hybrid reranking—not a complete error count for every method."
    )
    st.dataframe(failures, hide_index=True, use_container_width=True)

