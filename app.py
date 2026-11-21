import streamlit as st

st.set_page_config(
    page_title="Fraud Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.ui import apply_theme

apply_theme()

with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 4px 4px;">
          <div style="font-size:1.18rem;font-weight:800;color:#FFFFFF;letter-spacing:-0.02em;">
            Fraud Research
          </div>
          <div style="font-size:0.78rem;color:#A9BED4;margin-top:5px;">
            Evidence-grounded intelligence
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Hybrid retrieval · BM25 + FAISS + RRF")
    st.caption("Live generation · OpenRouter")

pages = {
    "Overview": [
        st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    ],
    "Research": [
        st.Page("pages/research_assistant.py", title="Research Assistant", icon="💬"),
    ],
    "Experiment Dashboard": [
        st.Page("pages/chunking_analysis.py", title="Chunking Analysis", icon="🧩"),
        st.Page("pages/retrieval_comparison.py", title="Retrieval Comparison", icon="🔍"),
        st.Page("pages/system_comparison.py", title="System Comparison", icon="📊"),
        st.Page("pages/error_analysis.py", title="Error Analysis", icon="🛠️"),
    ],
}

selected_page = st.navigation(pages)
selected_page.run()
