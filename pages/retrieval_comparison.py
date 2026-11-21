"""Retrieval Comparison dashboard from the 95-question Notebook 09 evaluation.

Quality metrics: 70 answerable questions with gold supporting evidence.
Latency measurements: all 95 questions.
No OpenRouter requests are made by this page.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from core.ui import apply_theme, page_header

apply_theme()

REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
SUMMARY_FILE = "article_page_metrics_summary.csv"
PER_QUESTION_FILE = "article_page_metrics_per_question.csv"
TIMING_FILE = "retrieval_timings.csv"

STRATEGIES = ("fixed_size", "sentence_based", "recursive", "section_aware")
METHODS = ("bm25", "dense_faiss", "hybrid_rrf", "hybrid_reranked")
STRATEGY_NAMES = {
    "fixed_size": "Fixed-size", "sentence_based": "Sentence-based",
    "recursive": "Recursive", "section_aware": "Section-aware",
}
METHOD_NAMES = {
    "bm25": "BM25", "dense_faiss": "Dense FAISS",
    "hybrid_rrf": "Hybrid RRF", "hybrid_reranked": "Hybrid + reranker",
}
METRIC_COLUMNS = {
    "Article Hit@K (passages)": "article_hit_at_{k}_passages",
    "Article Recall@K": "article_recall_at_{k}",
    "Article Precision@K": "article_precision_at_{k}",
    "Article MRR@K": "article_mrr_at_{k}",
    "Article nDCG@K": "article_ndcg_at_{k}",
    "Page-overlap Hit@K": "page_overlap_hit_at_{k}",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    """File mtime is included in the cache key so replacing a CSV refreshes data."""
    _ = modified_ns
    return pd.read_csv(path)


def load_report(filename: str) -> pd.DataFrame:
    path = REPORTS_DIR / filename
    return read_csv(str(path), path.stat().st_mtime_ns)


def download_table(frame: pd.DataFrame, filename: str, button_text: str) -> None:
    st.download_button(
        button_text, data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename, mime="text/csv",
    )


def format_method(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def format_strategy(strategy: str) -> str:
    return STRATEGY_NAMES.get(strategy, strategy)


page_header(
    "Retrieval Comparison",
    "Benchmark BM25, Dense FAISS, Hybrid RRF, and cross-encoder reranking using verified evaluation exports.",
    "EXPERIMENTAL DASHBOARD",
)
st.info(
    "**Evaluation scope:** Article-level and page-overlap retrieval metrics are computed "
    "using the **70 answerable questions** with gold evidence. Retrieval latency was "
    "measured on **all 95 questions**. These figures are *retrieval* measurements, "
    "not LLM-generated answer scores."
)

required = [SUMMARY_FILE, PER_QUESTION_FILE, TIMING_FILE]
missing = [name for name in required if not (REPORTS_DIR / name).is_file()]
if missing:
    st.error("Notebook 09 report files are missing from `data/reports/`.")
    st.code("\n".join(missing), language="text")
    st.stop()

try:
    summary = load_report(SUMMARY_FILE)
    per_question = load_report(PER_QUESTION_FILE)
    timings = load_report(TIMING_FILE)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Failed to read Notebook 09 CSV files: {exc}")
    st.stop()

all_metric_fields = {
    pattern.format(k=k)
    for pattern in METRIC_COLUMNS.values()
    for k in (1, 3, 5, 10)
}
needs = {
    SUMMARY_FILE: (summary, {"chunking_strategy", "method"} | all_metric_fields),
    PER_QUESTION_FILE: (
        per_question, {"question_id", "chunking_strategy", "method"} | all_metric_fields
    ),
    TIMING_FILE: (
        timings, {"question_id", "chunking_strategy", "method", "retrieval_latency_ms"}
    ),
}
for filename, (frame, expected_columns) in needs.items():
    absent = expected_columns.difference(frame.columns)
    if absent:
        st.error(f"`{filename}` lacks expected columns: {', '.join(sorted(absent))}")
        st.stop()

for label, frame, keys in (
    ("summary", summary, ["chunking_strategy", "method"]),
    ("per-question", per_question, ["question_id", "chunking_strategy", "method"]),
    ("timing", timings, ["question_id", "chunking_strategy", "method"]),
):
    if frame.duplicated(keys).any():
        st.error(f"Duplicate {label} records found; check your CSV exports.")
        st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Retrieval methods", summary["method"].nunique())
m2.metric("Chunking strategies", summary["chunking_strategy"].nunique())
m3.metric("Gold-evaluated questions", per_question["question_id"].nunique())
m4.metric("Timed questions", timings["question_id"].nunique())

st.divider()
col1, col2, col3 = st.columns([1.2, 0.7, 1.5])
with col1:
    strategy = st.selectbox(
        "Chunking strategy", STRATEGIES, index=3, format_func=format_strategy,
        help="Section-aware is the strategy used in your final A–E QA experiment.",
    )
with col2:
    k = st.selectbox("Top-K passages", (1, 3, 5, 10), index=2)
with col3:
    metric_name = st.selectbox("Retrieval metric", tuple(METRIC_COLUMNS), index=4)
metric_field = METRIC_COLUMNS[metric_name].format(k=k)

selected_summary = summary.loc[summary["chunking_strategy"].eq(strategy)].copy()
selected_summary[metric_field] = pd.to_numeric(
    selected_summary[metric_field], errors="coerce"
)
selected_summary["Method"] = selected_summary["method"].map(METHOD_NAMES)
selected_summary["Method"] = pd.Categorical(
    selected_summary["Method"],
    categories=[METHOD_NAMES[m] for m in METHODS], ordered=True,
)
selected_summary = selected_summary.sort_values("Method")

st.subheader(f"Retrieval methods — {format_strategy(strategy)}")
fig_main = px.bar(
    selected_summary, x="Method", y=metric_field, text=metric_field,
    category_orders={"Method": [METHOD_NAMES[m] for m in METHODS]},
    labels={metric_field: f"{metric_name} (0–1)"},
)
fig_main.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_main.update_layout(
    yaxis_range=[0, 1.10], margin=dict(t=15, b=15),
)
st.plotly_chart(fig_main, use_container_width=True)

valid = selected_summary.dropna(subset=[metric_field])
if not valid.empty:
    maximum = valid[metric_field].max()
    winners = valid.loc[valid[metric_field].eq(maximum), "Method"].astype(str).tolist()
    st.caption(
        f"Highest observed {metric_name} for {format_strategy(strategy)} "
        f"at K={k}: {', '.join(winners)} ({maximum:.3f}). "
        "This is a descriptive ranking, not proof of statistical superiority."
    )

st.subheader("Retrieval quality at multiple K values")
curve_rows = []
for method in METHODS:
    method_records = selected_summary.loc[selected_summary["method"].eq(method)]
    if method_records.empty:
        continue
    row = method_records.iloc[0]
    for top_k in (1, 3, 5, 10):
        field = METRIC_COLUMNS[metric_name].format(k=top_k)
        curve_rows.append({
            "Method": METHOD_NAMES[method], "Top K": top_k,
            "Score": pd.to_numeric(row[field], errors="coerce"),
        })
curves = pd.DataFrame(curve_rows)
fig_curve = px.line(
    curves, x="Top K", y="Score", color="Method", markers=True,
    category_orders={"Method": [METHOD_NAMES[m] for m in METHODS]},
    labels={"Score": f"{metric_name} (0–1)"},
)
fig_curve.update_xaxes(tickvals=[1, 3, 5, 10])
fig_curve.update_layout(yaxis_range=[0, 1.05], margin=dict(t=15, b=15))
st.plotly_chart(fig_curve, use_container_width=True)
st.caption(
    "Top-K counts retrieved *passages*. Metrics marked 'Article' assess supporting "
    "research articles; Page-overlap Hit@K checks supporting page overlap."
)

st.divider()
st.subheader("Cross-strategy retrieval comparison")
all_strategies = summary[["chunking_strategy", "method", metric_field]].copy()
all_strategies[metric_field] = pd.to_numeric(
    all_strategies[metric_field], errors="coerce"
)
all_strategies["Strategy"] = all_strategies["chunking_strategy"].map(STRATEGY_NAMES)
all_strategies["Method"] = all_strategies["method"].map(METHOD_NAMES)
heatmap = (
    all_strategies.pivot(index="Strategy", columns="Method", values=metric_field)
    .reindex(index=[STRATEGY_NAMES[s] for s in STRATEGIES],
             columns=[METHOD_NAMES[m] for m in METHODS])
)
fig_heatmap = px.imshow(
    heatmap, text_auto=".3f", zmin=0, zmax=1,
    labels={"x": "Retrieval method", "y": "Chunking strategy", "color": "Score"},
    aspect="auto",
)
fig_heatmap.update_layout(margin=dict(t=15, b=15))
st.plotly_chart(fig_heatmap, use_container_width=True)

show_metrics = [
    "article_hit_at_5_passages", "article_recall_at_5", "article_mrr_at_5",
    "article_ndcg_at_5", "page_overlap_hit_at_5",
]
summary_table = summary.loc[summary["chunking_strategy"].eq(strategy),
                            ["method"] + show_metrics].copy()
summary_table.insert(0, "Strategy", format_strategy(strategy))
summary_table["method"] = summary_table["method"].map(METHOD_NAMES)
summary_table.rename(columns={
    "method": "Method", "article_hit_at_5_passages": "Article Hit@5",
    "article_recall_at_5": "Article Recall@5",
    "article_mrr_at_5": "Article MRR@5",
    "article_ndcg_at_5": "Article nDCG@5",
    "page_overlap_hit_at_5": "Page-overlap Hit@5",
}, inplace=True)
st.subheader("Performance summary — K=5")
st.dataframe(summary_table, hide_index=True, use_container_width=True)
download_table(summary_table, "retrieval_comparison_top5.csv", "Download K=5 comparison")

st.divider()
st.subheader("Measured retrieval latency")
st.caption(
    "Timing is based on all 95 evaluation questions. `retrieval_latency_ms` is "
    "reported as recorded by Notebook 09. Query encoding is recorded separately "
    "when present; do not add the two times without checking the original timing definition."
)
timing_data = timings.loc[timings["chunking_strategy"].eq(strategy)].copy()
timing_data["retrieval_latency_ms"] = pd.to_numeric(
    timing_data["retrieval_latency_ms"], errors="coerce"
)
timing_data = timing_data.dropna(subset=["retrieval_latency_ms"])
latency_table = (
    timing_data.groupby("method", as_index=False)
    .agg(
        timed_questions=("question_id", "nunique"),
        mean_ms=("retrieval_latency_ms", "mean"),
        median_ms=("retrieval_latency_ms", "median"),
        p95_ms=("retrieval_latency_ms", lambda values: values.quantile(0.95)),
    )
)
latency_table["Method"] = latency_table["method"].map(METHOD_NAMES)
latency_table = latency_table.set_index("method").reindex(METHODS).reset_index()
latency_table["Method"] = latency_table["method"].map(METHOD_NAMES)
fig_latency = px.bar(
    latency_table, x="Method", y="mean_ms", text="mean_ms",
    category_orders={"Method": [METHOD_NAMES[m] for m in METHODS]},
    labels={"mean_ms": "Mean recorded retrieval latency (ms)"},
)
fig_latency.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_latency.update_layout(margin=dict(t=15, b=15))
st.plotly_chart(fig_latency, use_container_width=True)
st.dataframe(
    latency_table[["Method", "timed_questions", "mean_ms", "median_ms", "p95_ms"]]
    .rename(columns={
        "timed_questions": "Timed questions", "mean_ms": "Mean (ms)",
        "median_ms": "Median (ms)", "p95_ms": "P95 (ms)",
    }),
    hide_index=True, use_container_width=True,
)

st.divider()
st.subheader("Question-level retrieval results")
question_ids = sorted(per_question["question_id"].astype(str).unique())
qid = st.selectbox("Choose question (gold-evaluated answerable questions)", question_ids)
question_rows = per_question.loc[
    per_question["chunking_strategy"].eq(strategy)
    & per_question["question_id"].astype(str).eq(qid),
    ["method", "article_hit_at_5_passages", "article_recall_at_5",
     "article_ndcg_at_5", "page_overlap_hit_at_5"],
].copy()
question_rows["method"] = question_rows["method"].map(METHOD_NAMES)
question_rows.rename(columns={
    "method": "Method", "article_hit_at_5_passages": "Article Hit@5",
    "article_recall_at_5": "Article Recall@5",
    "article_ndcg_at_5": "Article nDCG@5",
    "page_overlap_hit_at_5": "Page-overlap Hit@5",
}, inplace=True)
st.dataframe(question_rows, hide_index=True, use_container_width=True)

st.caption(
    "**Experiment provenance:** All retrieval figures are from the saved "
    "Notebook 09 evaluation, not live OpenRouter API results. Final Systems A–E "
    "answer-generation scores are presented on the separate System Comparison page."
)
