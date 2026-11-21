"""Shared presentation helpers for the Fraud Research Assistant."""

from __future__ import annotations

import streamlit as st


PRIMARY = "#2358A3"
TEAL = "#0F766E"
NAVY = "#10233F"
PURPLE = "#6D5BD0"

PLOT_TEMPLATE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#1E293B"},
    "colorway": ["#2358A3", "#0F766E", "#6D5BD0", "#D97706", "#64748B"],
    "xaxis": {
        "gridcolor": "#E7EDF5",
        "linecolor": "#CBD5E1",
        "tickfont": {"color": "#475569"},
    },
    "yaxis": {
        "gridcolor": "#E7EDF5",
        "linecolor": "#CBD5E1",
        "tickfont": {"color": "#475569"},
    },
}


def apply_theme() -> None:
    """Apply a single responsive visual system to every page."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        :root {
          --navy: #10233F;
          --blue: #2358A3;
          --teal: #0F766E;
          --purple: #6D5BD0;
          --ink: #172033;
          --muted: #64748B;
          --line: #E4EAF2;
          --surface: #FFFFFF;
          --canvas: #F4F7FB;
        }

        html, body, [class*="css"] {
          font-family: "Inter", "Segoe UI", sans-serif;
        }
        .stApp {
          background: linear-gradient(135deg, #F8FAFD 0%, #F3F6FA 100%);
          color: var(--ink);
        }
        [data-testid="stHeader"] {
          background: rgba(248,250,253,0.82);
          backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #10233F 0%, #173555 100%);
          border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] * {
          color: #DCE7F4;
        }
        [data-testid="stSidebar"] hr {
          border-color: rgba(255,255,255,0.10);
        }
        [data-testid="stSidebar"] [data-testid="stLinkButton"],
        [data-testid="stSidebar"] .stButton button {
          color: #FFFFFF;
        }
        [data-testid="stSidebarNav"] a {
          border-radius: 10px;
          margin: 3px 6px;
          padding: 8px 10px !important;
        }
        [data-testid="stSidebarNav"] a:hover {
          background: rgba(255,255,255,0.08);
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
          background: linear-gradient(90deg, rgba(35,88,163,0.95), rgba(15,118,110,0.78));
          box-shadow: 0 7px 18px rgba(3,18,39,0.22);
        }

        .block-container {
          max-width: 1180px;
          padding-top: 2.2rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          color: var(--navy) !important;
          font-family: "Manrope", "Segoe UI", sans-serif;
          letter-spacing: -0.025em;
        }
        h1 { font-size: clamp(2rem, 3vw, 2.75rem); }
        h2 { font-size: 1.45rem; }
        h3 { font-size: 1.12rem; }
        p, li, label, [data-testid="stCaptionContainer"] {
          color: #4B5B70;
        }

        [data-testid="stMetric"] {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 18px 18px 15px;
          box-shadow: 0 8px 24px rgba(30,58,95,0.06);
        }
        [data-testid="stMetricLabel"] {
          color: #64748B;
          font-size: 0.78rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
          color: var(--navy);
          font-family: "Manrope", "Segoe UI", sans-serif;
          font-weight: 800;
        }

        [data-testid="stExpander"] {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 14px;
          overflow: hidden;
          box-shadow: 0 5px 18px rgba(30,58,95,0.04);
        }
        [data-testid="stExpander"] summary {
          font-weight: 700;
          color: var(--navy);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
          border: 1px solid var(--line);
          border-radius: 14px;
          overflow: hidden;
          background: white;
        }

        .stButton button, .stLinkButton a {
          border-radius: 10px;
          border: 1px solid #CAD5E4;
          font-weight: 700;
          transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }
        .stButton button:hover, .stLinkButton a:hover {
          transform: translateY(-1px);
          border-color: #91A7C4;
          box-shadow: 0 8px 18px rgba(35,88,163,0.10);
        }
        .stButton button[kind="primary"], [data-testid="baseButton-primary"] {
          background: linear-gradient(135deg, var(--blue), var(--teal));
          border: none;
          color: white;
        }

        [data-testid="stChatMessage"] {
          background: white;
          border: 1px solid var(--line);
          border-radius: 16px;
          box-shadow: 0 6px 20px rgba(30,58,95,0.05);
          margin: 10px 0;
        }
        [data-testid="stChatInput"] {
          border: 1px solid #CAD5E4;
          border-radius: 16px;
          box-shadow: 0 10px 28px rgba(30,58,95,0.10);
          background: white;
        }

        .fraud-page-header {
          background: linear-gradient(135deg, #10233F 0%, #1E4C7A 62%, #0F766E 100%);
          border-radius: 22px;
          padding: 28px 32px;
          color: white;
          box-shadow: 0 16px 38px rgba(16,35,63,0.16);
          margin-bottom: 20px;
        }
        .fraud-page-header h1,
        .fraud-page-header h2,
        .fraud-page-header p { color: white !important; }
        .fraud-page-header p { margin: 6px 0 0; opacity: 0.86; }

        .fraud-card {
          background: white;
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 20px;
          box-shadow: 0 8px 24px rgba(30,58,95,0.06);
          height: 100%;
        }
        .fraud-card h3 { margin-top: 0; }
        .fraud-card p { margin-bottom: 0; }

        .fraud-kicker {
          color: #7EA3CC;
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
        .fraud-accent-line {
          height: 3px;
          width: 44px;
          border-radius: 99px;
          background: linear-gradient(90deg, #2358A3, #0F766E);
          margin: 12px 0 16px;
        }

        @media (max-width: 700px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .fraud-page-header { padding: 22px 20px; border-radius: 16px; }
          [data-testid="stMetric"] { padding: 14px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str = "") -> None:
    """Render a consistent premium dashboard heading."""
    kicker_html = f'<div class="fraud-kicker">{kicker}</div>' if kicker else ""
    st.markdown(
        f"""
        <section class="fraud-page-header">
          {kicker_html}
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

