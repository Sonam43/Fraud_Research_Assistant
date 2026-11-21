"""Streamlit System Comparison page for Notebook 11 experiment outputs.

Place this file at pages/system_comparison.py in FraudResearchAssistant_Step1.
Requires the three Notebook 11 CSVs under data/reports/.
No API calls, index downloads, or reruns of the experiments.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from core.ui import apply_theme, page_header

apply_theme()

REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
REQUIRED_FILES = {
    "summary": "system_summary.csv",
    "by_type": "metrics_by_question_type.csv",
    "pairs": "paired_rag_comparisons.csv",
}
SYSTEM_ORDER = ["A", "B", "C", "D", "E"]
SYSTEM_NAMES = {
    "A": "A — No retrieval",
    "B": "B — BM25",
    "C": "C — Dense FAISS",
    "D": "D — Hybrid + reranker",
    "E": "E — Hybrid RRF",
}


@st.cache_data(show_spinner=False)
def load_report(path_str: str, mtime_ns: int) -> pd.DataFrame:
    """mtime makes the cache refresh when the user replaces an export."""
    _ = mtime_ns
    return pd.read_csv(path_str)


def get_data(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / REQUIRED_FILES[name]
    return load_report(str(path), path.stat().st_mtime_ns)


def download_csv(label: str, data: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


page_header(
    "System Comparison — A to E",
    "Review historical Ollama Qwen3:8B experiment results across five question-answering systems.",
    "EXPERIMENTAL DASHBOARD",
)
st.info(
    "The live application uses OpenRouter, but the scores on this page "
    "belong to the **earlier Qwen3:8B experiment**. They do not measure "
    "the deployed OpenRouter model."
)

missing = [
    name for name in REQUIRED_FILES.values() if not (REPORTS_DIR / name).is_file()
]
if missing:
    st.error("Missing dashboard CSV files in `data/reports/`:")
    st.code("\n".join(missing), language="text")
    st.stop()

try:
    summary = get_data("summary")
    by_type = get_data("by_type")
    paired = get_data("pairs")
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"Could not load one of the Notebook 11 exports: {exc}")
    st.stop()

required_summary = {
    "system", "n_questions", "n_answerable",
    "n_provisional_unanswerable", "mean_correctness_answerable_0_4",
    "mean_faithfulness_rag_answered_0_4",
    "mean_citation_support_rag_answered_0_4",
    "inappropriate_abstention_rate_answerable",
    "provisional_unanswerable_abstention_rate", "generation_latency_ms_mean",
}
required_type = {
    "system", "question_type", "n", "mean_answer_correctness_0_4",
    "mean_faithfulness_answered_0_4", "mean_citation_support_answered_0_4",
}
required_paired = {
    "first_system", "second_system", "n_answerable_questions",
    "mean_difference_second_minus_first", "ci95_low", "ci95_high",
    "sign_test_p_holm", "exploratory_holm_p_lt_0_05",
}
for name, frame, fields in [
    ("system_summary.csv", summary, required_summary),
    ("metrics_by_question_type.csv", by_type, required_type),
    ("paired_rag_comparisons.csv", paired, required_paired),
]:
    absent = sorted(fields - set(frame.columns))
    if absent:
        st.error(f"{name} is missing expected columns: {', '.join(absent)}")
        st.stop()

summary = summary.copy()
summary["system"] = summary["system"].astype(str)
summary = summary.set_index("system").reindex(SYSTEM_ORDER).reset_index()
by_type = by_type.copy()
by_type["system"] = by_type["system"].astype(str)
by_type["system_name"] = by_type["system"].map(SYSTEM_NAMES)

st.subheader("Experiment overview")
n_questions = int(summary["n_questions"].dropna().iloc[0])
n_answerable = int(summary["n_answerable"].dropna().iloc[0])
n_unanswerable = int(summary["n_provisional_unanswerable"].dropna().iloc[0])
best = summary.loc[summary["mean_correctness_answerable_0_4"].idxmax()]
a, b, c, d = st.columns(4)
a.metric("Questions", f"{n_questions}")
b.metric("Answerable questions", f"{n_answerable}")
c.metric("Provisionally unanswerable", f"{n_unanswerable}")
d.metric("Highest mean correctness", f"System {best['system']}",
         f"{best['mean_correctness_answerable_0_4']:.3f} / 4", delta_color="off")

st.subheader("Answer quality by system")
metric_names = {
    "Answer correctness": "mean_correctness_answerable_0_4",
    "Faithfulness (answered RAG only)": "mean_faithfulness_rag_answered_0_4",
    "Citation support (answered RAG only)": "mean_citation_support_rag_answered_0_4",
}
selected = st.selectbox("Choose a quality metric", list(metric_names), index=0)
metric_column = metric_names[selected]
plot = summary[["system", metric_column]].copy()
plot["system_name"] = plot["system"].map(SYSTEM_NAMES)
plot[metric_column] = pd.to_numeric(plot[metric_column], errors="coerce")
fig = px.bar(
    plot.dropna(subset=[metric_column]),
    x="system_name", y=metric_column,
    color="system_name", category_orders={"system_name": list(SYSTEM_NAMES.values())},
    text=metric_column, labels={"system_name": "System", metric_column: "Mean score (0–4)"},
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(showlegend=False, yaxis_range=[0, 4.3], margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Correctness is averaged over answerable questions; grounding metrics "
    "are averaged only for answered RAG responses. System A has no retrieved "
    "passages, so its faithfulness and citation-support scores are N/A. "
    "These are automated LLM-judge rubric scores, not verified accuracy percentages."
)

st.subheader("Abstention behavior")
rates = summary[["system", "inappropriate_abstention_rate_answerable",
                 "provisional_unanswerable_abstention_rate"]].copy()
rates["system_name"] = rates["system"].map(SYSTEM_NAMES)
rates = rates.melt(
    id_vars="system_name", value_vars=[
        "inappropriate_abstention_rate_answerable",
        "provisional_unanswerable_abstention_rate",
    ], var_name="measure", value_name="rate"
)
rates["measure"] = rates["measure"].map({
    "inappropriate_abstention_rate_answerable": "Abstains on answerable questions (lower is better)",
    "provisional_unanswerable_abstention_rate": "Abstains on provisional unanswerable questions (higher is better)",
})
rates["percentage"] = pd.to_numeric(rates["rate"], errors="coerce") * 100
fig2 = px.bar(
    rates, x="system_name", y="percentage", color="measure", barmode="group",
    category_orders={"system_name": list(SYSTEM_NAMES.values())},
    labels={"system_name": "System", "percentage": "Questions (%)", "measure": "Rate"},
)
fig2.update_layout(yaxis_range=[0, 105], margin=dict(t=20, b=20))
st.plotly_chart(fig2, use_container_width=True)
st.caption("The 25 unanswerable labels are provisional pending manual corpus-wide confirmation.")

st.subheader("Full system summary")
table_cols = {
    "system": "System",
    "n_answered": "Answered",
    "n_abstained": "Abstained",
    "mean_correctness_answerable_0_4": "Correctness / 4",
    "mean_faithfulness_rag_answered_0_4": "Faithfulness / 4",
    "mean_citation_support_rag_answered_0_4": "Citation support / 4",
    "inappropriate_abstention_rate_answerable": "Incorrect abstention rate",
    "provisional_unanswerable_abstention_rate": "Unanswerable abstention rate",
    "generation_latency_ms_mean": "Mean generation latency (ms)",
}
view_cols = [col for col in table_cols if col in summary.columns]
st.dataframe(summary[view_cols].rename(columns=table_cols), use_container_width=True, hide_index=True)
st.caption("Latency above covers answer generation; it is not necessarily end-to-end response time.")

st.subheader("Performance by question category")
category = st.selectbox("Question category", sorted(by_type["question_type"].dropna().unique()))
category_data = by_type[by_type["question_type"].eq(category)].copy()
category_data = category_data.set_index("system").reindex(SYSTEM_ORDER).reset_index()
category_data["system_name"] = category_data["system"].map(SYSTEM_NAMES)
st.dataframe(
    category_data[[
        "system_name", "n", "mean_answer_correctness_0_4",
        "mean_faithfulness_answered_0_4", "mean_citation_support_answered_0_4",
    ]].rename(columns={
        "system_name": "System", "n": "Questions",
        "mean_answer_correctness_0_4": "Correctness / 4",
        "mean_faithfulness_answered_0_4": "Faithfulness / 4",
        "mean_citation_support_answered_0_4": "Citation support / 4",
    }),
    use_container_width=True, hide_index=True,
)

st.subheader("Paired statistical comparisons (RAG systems)")
st.caption(
    "Paired differences are calculated on the same answerable questions. "
    "Positive values favor the second system. Holm-adjusted p-values account "
    "for multiple comparisons; these tests are exploratory."
)
pair_view = paired[[
    "first_system", "second_system", "n_answerable_questions",
    "mean_difference_second_minus_first", "ci95_low", "ci95_high",
    "sign_test_p_holm", "exploratory_holm_p_lt_0_05",
]].rename(columns={
    "first_system": "First", "second_system": "Second",
    "n_answerable_questions": "Paired questions",
    "mean_difference_second_minus_first": "Second − first",
    "ci95_low": "95% CI lower", "ci95_high": "95% CI upper",
    "sign_test_p_holm": "Holm-adjusted p-value",
    "exploratory_holm_p_lt_0_05": "Exploratory p < 0.05",
})
st.dataframe(pair_view, use_container_width=True, hide_index=True)

with st.expander("Download research tables"):
    download_csv("Download system summary", summary, "system_summary.csv")
    download_csv("Download category metrics", by_type.drop(columns="system_name"), "metrics_by_question_type.csv")
    download_csv("Download paired comparisons", paired, "paired_rag_comparisons.csv")

st.caption(
    "Source: DAT401 financial-fraud-detection research QA experiment, "
    "Notebook 11 (Qwen3:8B generation and evaluation). "
    "A deployed OpenRouter generator should be evaluated separately."
)
