import streamlit as st
from core.config import get_setting, openrouter_is_configured
from core.llm_client import test_connection
from core.ui import apply_theme
import pandas as pd
from pathlib import Path

apply_theme()

st.markdown(
    """
    <section class="fraud-page-header" style="padding:38px 42px;">
      <div class="fraud-kicker">AI RESEARCH PLATFORM · DAT401</div>
      <h1 style="font-size:clamp(2.25rem,4vw,3.4rem);max-width:780px;margin:8px 0;">
        Financial Fraud Research Assistant
      </h1>
      <p style="font-size:1.05rem;max-width:760px;">
        Evidence-grounded literature intelligence for financial fraud detection research,
        combining section-aware retrieval, hybrid ranking, and verifiable citations.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.7, 1])
with hero_left:
    st.page_link("pages/research_assistant.py", label="Start researching", icon="💬", use_container_width=True)
with hero_right:
    if openrouter_is_configured():
        st.success("OpenRouter connected", icon="✅")
    else:
        st.warning("OpenRouter key not configured", icon="⚠️")

st.markdown("### Research platform at a glance")
c1, c2, c3, c4 = st.columns(4)

project_root = Path(__file__).resolve().parents[1]
inventory = project_root / "data" / "metadata" / "corpus_inventory.csv"
paper_count = len(pd.read_csv(inventory)) if inventory.is_file() else 0
c1.metric("Research questions", "95")
c2.metric("Answerable questions", "70")
c3.metric("Corpus papers", f"{paper_count}" if paper_count else "Not loaded")
c4.metric("QA systems evaluated", "5")

st.markdown("### Explore the platform")
features = st.columns(4)
feature_data = [
    ("01", "Hybrid retrieval", "BM25 and dense FAISS results are fused with Reciprocal Rank Fusion."),
    ("02", "AI answers", "OpenRouter generation produces concise, evidence-grounded responses."),
    ("03", "Research citations", "Inline citations link to indexed passages, page numbers, and source metadata."),
    ("04", "Verified evaluation", "Interactive dashboards report actual saved Notebook 09 and 11 results."),
]
for column, (number, title, body) in zip(features, feature_data):
    with column:
        st.markdown(
            f'<div class="fraud-card"><div class="fraud-kicker">{number}</div>'
            f'<h3>{title}</h3><div class="fraud-accent-line"></div><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("### Research pipeline")
st.markdown(
    """
    <div style="display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:10px 0 22px;">
      <div class="fraud-card" style="text-align:center;padding:14px 8px;"><b>Research papers</b></div>
      <div class="fraud-card" style="text-align:center;padding:14px 8px;"><b>Section-aware chunks</b></div>
      <div class="fraud-card" style="text-align:center;padding:14px 8px;"><b>Embeddings</b></div>
      <div class="fraud-card" style="text-align:center;padding:14px 8px;"><b>Hybrid retrieval</b></div>
      <div class="fraud-card" style="text-align:center;padding:14px 8px;"><b>OpenRouter LLM</b></div>
      <div class="fraud-card" style="text-align:center;padding:14px 8px;"><b>Evidence-grounded answer</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Project workspaces")
n1, n2, n3, n4, n5 = st.columns(5)
nav_items = [
    (n1, "pages/research_assistant.py", "Research Assistant", "Ask evidence-grounded questions"),
    (n2, "pages/chunking_analysis.py", "Chunking", "Compare four chunking strategies"),
    (n3, "pages/retrieval_comparison.py", "Retrieval", "Compare retrieval methods"),
    (n4, "pages/system_comparison.py", "Systems A–E", "Compare QA experiments"),
    (n5, "pages/error_analysis.py", "Error Analysis", "Inspect limitations and failures"),
]
for column, page, title, caption in nav_items:
    with column:
        st.markdown(
            f'<div class="fraud-card"><h3 style="font-size:1rem;">{title}</h3>'
            f'<p style="min-height:48px;">{caption}</p></div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label="Open", use_container_width=True)

st.divider()
with st.expander("OpenRouter configuration"):
    if openrouter_is_configured():
        st.success("An OpenRouter API key is configured; the key is not displayed.")
    else:
        st.warning("No OpenRouter API key found. Create `.streamlit/secrets.toml` from the example file.")
    st.write(f"Model setting: `{get_setting('OPENROUTER_MODEL', 'openrouter/free')}`")
    st.caption("The connectivity test sends one API request and may count toward your quota.")
    if st.button("Test OpenRouter connection", disabled=not openrouter_is_configured()):
        with st.spinner("Checking hosted model access..."):
            try:
                model, response = test_connection()
                st.success(f"Request succeeded with model setting: {model}")
                st.code(response)
            except Exception as exc:
                st.error(
                    f"Connection failed ({type(exc).__name__}). "
                    "Check the model ID, API balance, and account limits."
                )
