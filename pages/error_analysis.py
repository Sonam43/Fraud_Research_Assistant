"""Notebook 11 error-analysis dashboard for the historical Qwen3:8B experiment.

This page uses only local CSV files and never makes an API request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from core.ui import apply_theme, page_header

apply_theme()

REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
REQUIRED_FILES = {
    "casebook": "error_casebook.csv",
    "taxonomy": "error_taxonomy_counts.csv",
    "citations": "citation_structure_audit.csv",
    "details": "error_case_details.csv",
}
SYSTEM_ORDER = ["A", "B", "C", "D", "E"]
SYSTEM_LABELS = {
    "A": "A — No retrieval",
    "B": "B — BM25",
    "C": "C — Dense FAISS",
    "D": "D — Hybrid + reranker",
    "E": "E — Hybrid RRF",
}
FLAG_NAMES = {
    "unnecessary_abstention_on_answerable": "Abstained on an answerable question",
    "low_faithfulness_judge_score": "Low faithfulness judge score",
    "low_citation_support_judge_score": "Low citation-support judge score",
    "low_answer_correctness": "Low answer-correctness judge score",
    "answered_provisionally_unanswerable": "Answered a provisionally unanswerable question",
}
CITATION_CHECKS = {
    "has_inline_marker_if_needed": "Answer includes inline citations",
    "inline_matches_listed_if_needed": "Inline and listed IDs agree",
    "all_ids_in_context_if_needed": "Cited IDs occur in context",
    "listed_matches_cited_metadata_if_needed": "Listed IDs match citation metadata",
}

@st.cache_data(show_spinner=False)
def read_report(path: str, mtime_ns: int) -> pd.DataFrame:
    """Reload CSV whenever the user replaces its file."""
    _ = mtime_ns
    return pd.read_csv(path)


def load_report(name: str) -> pd.DataFrame:
    file = REPORTS_DIR / REQUIRED_FILES[name]
    return read_report(str(file), file.stat().st_mtime_ns)


def flag_list(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def safe_json_list(value: object) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def display_score(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f} / 4"


page_header(
    "Error Analysis — Systems A–E",
    "Explore observed weaknesses, abstention behavior, citation issues, and judge-reported limitations.",
    "EXPERIMENTAL DASHBOARD",
)
st.info(
    "This dashboard analyzes your **historical Qwen3:8B experiment**, not your live "
    "OpenRouter generator. Error flags are screening rules based on dataset labels "
    "and automated judge scores; human verification is still necessary."
)

missing = [p for p in REQUIRED_FILES.values() if not (REPORTS_DIR / p).is_file()]
if missing:
    st.error("Missing files in `data/reports/`:")
    st.code("\n".join(missing), language="text")
    st.stop()

try:
    casebook = load_report("casebook")
    taxonomy = load_report("taxonomy")
    citations = load_report("citations")
    details = load_report("details")
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Cannot load Notebook 11 CSV results: {exc}")
    st.stop()

requirements = {
    "error_casebook.csv": (casebook, {
        "question_id", "system", "question_type", "answerable", "status",
        "correctness_0_4", "faithfulness_0_4", "citation_support_0_4",
        "issue_count", "issue_flags",
    }),
    "error_taxonomy_counts.csv": (taxonomy, {"flag", "n_question_system_pairs"}),
    "citation_structure_audit.csv": (citations, {"question_id", "system", "rag_answered", *CITATION_CHECKS}),
    "error_case_details.csv": (details, {
        "question_id", "system", "question", "answer", "reference_answer",
        "judge_reason", "judge_critical_issues_json", "citations_json",
        "citation_bibliography_json", "generation_latency_ms",
    }),
}
for filename, (df, expected) in requirements.items():
    absent = sorted(expected - set(df.columns))
    if absent:
        st.error(f"`{filename}` lacks required columns: {', '.join(absent)}")
        st.stop()
    if df.duplicated(["question_id", "system"]).any() if {"question_id", "system"}.issubset(df.columns) else False:
        st.error(f"`{filename}` has duplicate question–system pairs")
        st.stop()

merged = casebook.merge(
    details, on=["question_id", "system"], validate="one_to_one", how="left"
).merge(
    citations, on=["question_id", "system"], validate="one_to_one", how="left"
)
if len(merged) != len(casebook):
    st.error("A CSV merge changed the number of rows; verify your exports.")
    st.stop()

merged["system"] = merged["system"].astype(str)
merged["question_id"] = merged["question_id"].astype(str)
merged["is_answerable"] = merged["answerable"].map(is_true)
merged["flags"] = merged["issue_flags"].map(flag_list)
merged["flagged"] = merged["flags"].map(bool)
merged["issue_count"] = pd.to_numeric(merged["issue_count"], errors="coerce").fillna(0).astype(int)

st.subheader("Experiment-wide overview")
raw_flags = sorted({f for row in merged["flags"] for f in row})
n_issues = int(merged["flagged"].sum())
all_a, all_b, all_c = st.columns(3)
all_a.metric("Question–system pairs", f"{len(merged):,}")
all_b.metric("Pairs with ≥1 flag", f"{n_issues:,}")
all_c.metric("Total flag assignments", f"{sum(map(len, merged['flags'])):,}")
st.caption("One answer may trigger several flags, so flag counts cannot be added to obtain distinct affected answers.")

with st.expander("Published Notebook 11 flag definitions and counts"):
    global_counts = taxonomy.copy()
    global_counts["Issue category"] = global_counts["flag"].map(FLAG_NAMES).fillna(global_counts["flag"])
    st.dataframe(global_counts[["Issue category", "n_question_system_pairs"]].rename(
        columns={"n_question_system_pairs": "Flagged question–system pairs"}),
        use_container_width=True, hide_index=True,
    )

st.subheader("Explore errors")
filters1, filters2 = st.columns(2)
with filters1:
    selected_systems = st.multiselect(
        "Systems", SYSTEM_ORDER, default=SYSTEM_ORDER,
        format_func=lambda x: SYSTEM_LABELS.get(x, x),
    )
with filters2:
    selected_types = st.multiselect(
        "Question categories", sorted(merged["question_type"].dropna().unique().tolist()),
        default=sorted(merged["question_type"].dropna().unique().tolist()),
    )
filters3, filters4 = st.columns(2)
with filters3:
    answerability = st.selectbox(
        "Answerability label", ["All", "Answerable", "Provisionally unanswerable"]
    )
with filters4:
    flagged_only = st.checkbox("Show only flagged cases", value=True)

selected_flags = st.multiselect(
    "Issue flags (select any)", raw_flags,
    format_func=lambda f: FLAG_NAMES.get(f, f),
)

subset = merged[
    merged["system"].isin(selected_systems)
    & merged["question_type"].isin(selected_types)
].copy()
if answerability == "Answerable":
    subset = subset[subset["is_answerable"]]
elif answerability == "Provisionally unanswerable":
    subset = subset[~subset["is_answerable"]]
if flagged_only:
    subset = subset[subset["flagged"]]
if selected_flags:
    selected_flags_set = set(selected_flags)
    subset = subset[subset["flags"].map(lambda flags: bool(selected_flags_set.intersection(flags)))]

left, mid, right = st.columns(3)
left.metric("Matching cases", len(subset))
mid.metric("Distinct questions", int(subset["question_id"].nunique()))
right.metric("Flags in matching cases", int(sum(map(len, subset["flags"]))))

# A question can count in multiple issue classes.
counts = pd.Series([f for fs in subset["flags"] for f in fs], dtype="object").value_counts()
if not counts.empty:
    counts_df = counts.rename_axis("flag").reset_index(name="Cases")
    counts_df["Issue category"] = counts_df["flag"].map(FLAG_NAMES).fillna(counts_df["flag"])
    fig = px.bar(
        counts_df, x="Cases", y="Issue category", orientation="h",
        text="Cases", labels={"Issue category": "Error flag"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No flagged cases match the current filters. Clear a filter or disable 'Show only flagged cases'.")

view = subset.copy()
view["system_name"] = view["system"].map(SYSTEM_LABELS)
view["Issue flags"] = view["flags"].map(lambda values: "; ".join(FLAG_NAMES.get(f, f) for f in values))
view = view.sort_values(["issue_count", "question_id", "system"], ascending=[False, True, True])
view_cols = {
    "question_id": "Question ID",
    "system_name": "System",
    "question_type": "Category",
    "status": "Generation status",
    "issue_count": "Flags",
    "correctness_0_4": "Correctness /4",
    "faithfulness_0_4": "Faithfulness /4",
    "citation_support_0_4": "Citation support /4",
    "Issue flags": "Identified issues",
}
st.dataframe(view[list(view_cols)].rename(columns=view_cols), use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered error cases (CSV)",
    data=view[["question_id", "system", "question_type", "status", "issue_count", "Issue flags",
               "correctness_0_4", "faithfulness_0_4", "citation_support_0_4"]].to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_error_cases.csv", mime="text/csv",
)

st.subheader("Inspect an individual answer")
if view.empty:
    st.caption("Choose broader filters to inspect a generated answer.")
else:
    options = [f"{r.question_id} | System {r.system}" for r in view.itertuples(index=False)]
    selected = st.selectbox("Question and system", options)
    row = view.iloc[options.index(selected)]
    st.markdown(f"**Question ({row['question_type']}):**")
    st.write(row["question"] if pd.notna(row["question"]) else "Not available")
    a, b, c = st.columns(3)
    a.metric("Correctness", display_score(row["correctness_0_4"]))
    b.metric("Faithfulness", display_score(row["faithfulness_0_4"]))
    c.metric("Citation support", display_score(row["citation_support_0_4"]))
    st.markdown("**Generated answer**")
    st.write(row["answer"] if pd.notna(row["answer"]) else "Not available")
    with st.expander("Reference answer and judge explanation", expanded=True):
        st.markdown("**Reference answer (dataset)**")
        st.write(row["reference_answer"] if pd.notna(row["reference_answer"]) else "Not available")
        st.markdown("**Automated judge's reason**")
        st.write(row["judge_reason"] if pd.notna(row["judge_reason"]) else "Not available")
        issues = safe_json_list(row["judge_critical_issues_json"])
        if issues:
            st.markdown("**Judge-reported critical issues**")
            for issue in issues:
                st.write(f"• {issue}")
    with st.expander("Citation identifiers and source metadata"):
        cited_ids = safe_json_list(row["citations_json"])
        st.write("Cited source identifiers:", ", ".join(map(str, cited_ids)) or "None")
        bibliography = safe_json_list(row["citation_bibliography_json"])
        if bibliography:
            st.dataframe(pd.DataFrame(bibliography), use_container_width=True, hide_index=True)
        else:
            st.caption("No cited sources for this response.")
        st.caption("Citation IDs and paper titles are structural metadata; these checks do not prove factual support.")

st.subheader("Citation structure audit")
rag = citations[citations["rag_answered"].map(is_true)].copy()
check_rows = []
for column, title in CITATION_CHECKS.items():
    good = int(rag[column].map(is_true).sum())
    check_rows.append({
        "Check": title,
        "Passed": good,
        "Not passed / missing": len(rag)-good,
        "Applicable RAG answers": len(rag),
    })
checks_table = pd.DataFrame(check_rows)
st.dataframe(checks_table, use_container_width=True, hide_index=True)
st.caption(
    "A structurally valid citation links to a supplied passage ID. "
    "Human review must still assess whether the passage supports the specific claim."
)

st.caption("Source: Notebook 11 exported analysis files; the live OpenRouter endpoint is never called by this page.")
