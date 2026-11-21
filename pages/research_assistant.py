
"""
Financial Fraud Research Assistant
==================================

Features:
- Hybrid retrieval: BM25 + FAISS + RRF
- OpenRouter-based research question answering
- Concise, evidence-grounded answers
- Inline citations such as [S1] and [S2]
- Authors, publication years, venues, and DOI/URLs
- PDF page numbers and supporting passages
- Session-based chat history
- No unrelated evidence displayed for abstained answers
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

import streamlit as st

from core.rag_pipeline import answer_question
from core.retriever import HybridRetriever
from core.ui import apply_theme, page_header

apply_theme()


# ============================================================
# 1. CONFIGURATION
# ============================================================

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "corpus_inventory.csv"
)

CHAT_KEY = "fraud_research_chat_history"


# ============================================================
# 2. LOAD CORPUS INVENTORY
# ============================================================

@st.cache_data(show_spinner=False)
def load_bibliography() -> dict[str, dict[str, str]]:
    """
    Load bibliographic metadata from corpus_inventory.csv.

    Match retrieved papers using article_id.
    """

    if not INVENTORY_PATH.is_file():
        return {}

    bibliography = {}

    with INVENTORY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "article_id",
            "title",
            "author",
            "year",
            "venue",
            "doi_or_url",
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                "Missing corpus inventory columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            article_id = str(
                row.get("article_id") or ""
            ).strip().upper()

            if not article_id:
                continue

            if article_id in bibliography:
                raise ValueError(
                    f"Duplicate article_id: {article_id}"
                )

            bibliography[article_id] = row

    return bibliography


# ============================================================
# 3. LOAD HYBRID RETRIEVER
# ============================================================

@st.cache_resource(
    show_spinner="Loading FAISS and BM25 indexes..."
)
def get_retriever() -> HybridRetriever:
    """
    Load the verified section-aware retrieval indexes
    once for the Streamlit application.
    """

    return HybridRetriever()


# ============================================================
# 4. PROCESS DOI / URL
# ============================================================

def get_paper_link(
    doi_or_url: str
) -> tuple[str, str] | None:
    """
    Return a display label and a usable URL.

    Supports:
    - DOI URLs
    - Raw DOI identifiers
    - Ordinary HTTP/HTTPS URLs

    Does not invent missing identifiers.
    """

    value = doi_or_url.strip()

    if not value:
        return None

    if value.startswith(
        ("https://doi.org/", "http://doi.org/")
    ):

        doi = value.split("doi.org/", 1)[-1]

        return (
            f"DOI: {doi}",
            value
        )

    if re.match(
        r"^10\.\d{4,9}/\S+$",
        value
    ):

        return (
            f"DOI: {value}",
            f"https://doi.org/{value}"
        )

    if value.startswith(
        ("https://", "http://")
    ):

        return (
            "Research paper URL",
            value
        )

    return None


# ============================================================
# 5. DISPLAY RESEARCH PAPER DETAILS
# ============================================================

def show_source(
    source: dict[str, Any],
    *,
    show_excerpt: bool = True
) -> None:
    """
    Display one academic research source.

    Bibliographic information:
        corpus_inventory.csv

    Passage information:
        Verified hybrid retriever
    """

    # --------------------------------------------------------
    # 5.1 Extract retrieval metadata
    # --------------------------------------------------------

    source_id = str(
        source.get("source_id") or "Unknown"
    )

    article_id = str(
        source.get("article_id") or ""
    ).strip().upper()

    title = str(
        source.get("title")
        or "Untitled research paper"
    )

    pages = source.get("page_numbers") or []

    pages_text = (
        ", ".join(str(p) for p in pages)
        if pages
        else "Not recorded"
    )

    chunk_id = str(
        source.get("chunk_id")
        or "Not recorded"
    )

    # --------------------------------------------------------
    # 5.2 Retrieve corpus bibliography
    # --------------------------------------------------------

    record = load_bibliography().get(
        article_id,
        {}
    )

    if record.get("title"):
        title = str(
            record["title"]
        ).strip()

    authors = str(
        record.get("author") or ""
    ).strip()

    year = str(
        record.get("year") or ""
    ).strip()

    venue = str(
        record.get("venue") or ""
    ).strip()

    doi_or_url = str(
        record.get("doi_or_url") or ""
    ).strip()

    # --------------------------------------------------------
    # 5.3 Display reference
    # --------------------------------------------------------

    st.markdown(
        f"**[{source_id}] {title}**"
    )

    st.caption(
        f"Article ID: {article_id or 'Not recorded'}"
    )

    if authors:
        st.write(f"**Authors:** {authors}")

    if year:
        st.write(
            f"**Publication year:** {year}"
        )

    if venue:
        st.write(
            f"**Publication venue:** {venue}"
        )

    st.write(
        f"**Indexed PDF page(s):** {pages_text}"
    )

    # Display section heading only if it exists.
    section = (
        source.get("section_title")
        or source.get("section_heading")
        or source.get("section")
    )

    if section:
        st.write(
            f"**Section / Topic:** {section}"
        )

    # --------------------------------------------------------
    # 5.4 Display DOI / URL
    # --------------------------------------------------------

    paper_link = get_paper_link(
        doi_or_url
    )

    if paper_link:

        label, url = paper_link

        st.write(f"**{label}**")

        st.link_button(
            "Open research paper",
            url
        )

    elif doi_or_url:

        st.caption(
            f"Reference identifier: {doi_or_url}"
        )

    else:

        st.caption(
            "No DOI or research URL recorded."
        )

    # --------------------------------------------------------
    # 5.5 Display supporting passage
    # --------------------------------------------------------

    st.caption(
        f"Passage ID: {chunk_id}"
    )

    if show_excerpt:

        excerpt = str(
            source.get("excerpt")
            or "No supporting passage available."
        )

        st.markdown(
            "**Supporting research passage:**"
        )

        st.write(excerpt)

    st.divider()


# ============================================================
# 6. DISPLAY GENERATED ANSWER
# ============================================================

def show_answer(
    result: dict[str, Any]
) -> None:
    """
    Display the generated answer and research references.

    IMPORTANT:
    - Answered: show citations and evidence.
    - Insufficient evidence: show only abstention message.
    - Citation failure: show only warning.
    - Runtime error: show only error.

    The model is not called again when displaying history.
    """

    status = str(
        result.get("status") or "unknown"
    )

    answer = str(
        result.get("answer")
        or "No answer was returned."
    )

    # --------------------------------------------------------
    # 6.1 Insufficient evidence
    # --------------------------------------------------------

    if status == "insufficient_evidence":

        st.info(answer)

        # Do not show potentially unrelated passages.
        return

    # --------------------------------------------------------
    # 6.2 Citation validation failure
    # --------------------------------------------------------

    if status == "citation_validation_failed":

        st.warning(answer)

        st.caption(
            "The generated answer did not pass "
            "citation validation."
        )

        return

    # --------------------------------------------------------
    # 6.3 Other errors
    # --------------------------------------------------------

    if status != "answered":

        st.error(answer)

        return

    # --------------------------------------------------------
    # 6.4 Display grounded answer
    # --------------------------------------------------------

    st.markdown(answer)

    model = result.get("model")

    cited_sources = (
        result.get("sources") or []
    )

    retrieved_sources = (
        result.get("retrieved_sources") or []
    )

    if model:

        st.caption(
            f"Model: {model} | "
            f"Retrieved passages: {len(retrieved_sources)} | "
            f"Cited passages: {len(cited_sources)}"
        )

    # --------------------------------------------------------
    # 6.5 Cited research papers
    # --------------------------------------------------------

    if cited_sources:

        with st.expander(
            f"📚 Cited research papers ({len(cited_sources)})",
            expanded=False
        ):

            for source in cited_sources:

                show_source(
                    source,
                    show_excerpt=True
                )

    # --------------------------------------------------------
    # 6.6 All retrieved evidence
    # --------------------------------------------------------

    # This section only runs for answered questions
    # because other statuses returned earlier.

    if retrieved_sources:

        with st.expander(
            f"🔎 All retrieved passages ({len(retrieved_sources)})",
            expanded=False
        ):

            st.caption(
                "These passages were retrieved from "
                "the research corpus. Some passages "
                "may not have been cited in the answer."
            )

            for source in retrieved_sources:

                show_source(
                    source,
                    show_excerpt=True
                )

    # --------------------------------------------------------
    # 6.7 Citation verification note
    # --------------------------------------------------------

    st.caption(
        "Citation IDs were checked against retrieved "
        "passages. Individual claims should still be "
        "reviewed for factual support."
    )


# ============================================================
# 7. STREAMLIT PAGE HEADER
# ============================================================

page_header(
    "Financial Fraud Research Assistant",
    "Ask focused questions across your indexed research corpus. Answers are grounded in retrieved passages and validated citations.",
    "LIVE RESEARCH WORKSPACE",
)


# ============================================================
# 8. VERIFY CORPUS INVENTORY
# ============================================================

if INVENTORY_PATH.is_file():

    try:

        bibliography = load_bibliography()

        st.caption(
            f"Bibliographic metadata: "
            f"{len(bibliography)} research paper records loaded."
        )

    except Exception as exc:

        LOGGER.exception(
            "Failed to load corpus inventory."
        )

        st.warning(
            "Could not load corpus inventory: "
            f"{type(exc).__name__}. "
            "Check the VS Code terminal."
        )

else:

    st.warning(
        "corpus_inventory.csv was not found. "
        "Place it inside data/metadata/."
    )


# ============================================================
# 9. RESEARCH ASSISTANT SETTINGS
# ============================================================

with st.sidebar:

    st.markdown(
        "<div style='font-weight:800;font-size:1rem;margin:6px 0 10px;'>Research controls</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Adjust evidence depth and manage this session."
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help=(
            "How many research passages should "
            "be supplied to the LLM?"
        ),
    )

    if st.button(
        "Clear chat history",
        use_container_width=True
        , type="primary"
    ):

        st.session_state[CHAT_KEY] = []

        st.rerun()

    st.caption(
        "Chat history is stored within this session. "
        "Each question is currently answered independently."
    )


# ============================================================
# 10. EXPLAIN CITATIONS
# ============================================================

with st.expander(
    "How to interpret research citations",
    expanded=False
):

    st.write(
        "Citations such as [S1] and [S2] refer to "
        "research passages retrieved from your corpus."
    )

    st.write(
        "Expand 'Cited research papers' to see "
        "the authors, publication year, DOI/URL, "
        "PDF page numbers and supporting passages."
    )

    st.caption(
        "Bibliographic details come from "
        "corpus_inventory.csv. Passage and PDF page "
        "information come from the verified indexes."
    )


# ============================================================
# 11. INITIALIZE CHAT HISTORY
# ============================================================

if CHAT_KEY not in st.session_state:

    st.session_state[CHAT_KEY] = []

if not st.session_state[CHAT_KEY]:
    st.markdown("#### Suggested research questions")
    s1, s2, s3 = st.columns(3)
    suggestions = [
        (
            s1,
            "What machine learning models are commonly used for credit card fraud detection?",
        ),
        (
            s2,
            "How do researchers address severe class imbalance in fraud datasets?",
        ),
        (
            s3,
            "Which evaluation metrics are most appropriate for fraud detection?",
        ),
    ]
    for column, prompt in suggestions:
        if column.button(prompt, use_container_width=True):
            st.session_state["_suggested_research_question"] = prompt
            st.rerun()


# ============================================================
# 12. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state[CHAT_KEY]:

    with st.chat_message(
        message["role"]
    ):

        if message["role"] == "user":

            st.markdown(
                message["content"]
            )

        else:

            show_answer(
                message["result"]
            )


# ============================================================
# 13. ACCEPT RESEARCH QUESTION
# ============================================================

suggested_question = st.session_state.pop(
    "_suggested_research_question",
    "",
)
typed_question = st.chat_input(
    "Ask a question about financial fraud detection..."
)
question = suggested_question or typed_question


# ============================================================
# 14. GENERATE ANSWER
# ============================================================

if question and question.strip():

    question = question.strip()

    # --------------------------------------------------------
    # 14.1 Save the user question
    # --------------------------------------------------------

    st.session_state[CHAT_KEY].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # 14.2 Generate research answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Retrieving evidence and generating an answer..."
            ):

                result = answer_question(
                    question,
                    retriever=get_retriever(),
                    top_k=top_k,
                )

        except Exception as exc:

            LOGGER.exception(
                "Research Assistant request failed."
            )

            result = {
                "status": "runtime_error",
                "answer": (
                    "The request could not be completed. "
                    f"Error type: {type(exc).__name__}. "
                    "Check the VS Code terminal, "
                    "network connection and OpenRouter settings."
                ),
                "model": None,
                "sources": [],
                "retrieved_sources": [],
            }

        # ----------------------------------------------------
        # 14.3 Display the response
        # ----------------------------------------------------

        show_answer(result)

        # ----------------------------------------------------
        # 14.4 Save the response to chat history
        # ----------------------------------------------------

        st.session_state[CHAT_KEY].append({
            "role": "assistant",
            "result": result,
        })
