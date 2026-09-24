import streamlit as st

st.set_page_config(
    page_title="Web App for Exploratory Data Analysis and ML Workflows",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global styling — matches the ML/EDA + NLP dashboards
# (Playfair Display headings, DM Sans body, warm off-white background,
# colored stat/nav cards, dark active-tab treatment)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .stApp {
            background-color: #FAF6EF;
        }
        #MainMenu, footer, header {visibility: hidden;}

        .hero-title {
            font-family: 'Playfair Display', serif;
            font-weight: 800;
            font-size: 3rem;
            line-height: 1.15;
            color: #1A1A1A;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.05rem;
            color: #6B6B6B;
            margin-bottom: 0.25rem;
        }
        .hero-tags {
            font-size: 0.95rem;
            color: #9A9A9A;
            margin-bottom: 2.2rem;
        }
        .hero-divider {
            border: none;
            border-top: 1px solid #E5DFD3;
            margin: 1.6rem 0 2.4rem 0;
        }

        .path-card {
            background: #FFFFFF;
            border: 1px solid #ECE6D8;
            border-radius: 16px;
            padding: 2rem 1.9rem 1.7rem 1.9rem;
            height: 100%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .path-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 22px rgba(0,0,0,0.07);
        }
        .path-icon {
            font-size: 2.2rem;
            width: 56px;
            height: 56px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.1rem;
        }
        .icon-structured { background: #EAF1FB; }
        .icon-unstructured { background: #FBEAF5; }

        .path-title {
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            font-size: 1.5rem;
            color: #1A1A1A;
            margin-bottom: 0.5rem;
        }
        .path-desc {
            color: #6B6B6B;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 1.1rem;
        }
        .path-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 1.4rem;
        }
        .path-chip {
            background: #F5F1E8;
            color: #7A7256;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            border: 1px solid #ECE6D8;
        }
        .chip-blue { background: #EAF1FB; color: #3B5B8C; border-color: #DCE7F7; }
        .chip-pink { background: #FBEAF5; color: #9C3E77; border-color: #F5DDEC; }

        div.stLinkButton > a {
            background-color: #1A1A1A !important;
            color: #FAF6EF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.55rem 1rem !important;
            border: none !important;
            width: 100%;
            text-align: center;
        }
        div.stLinkButton > a:hover {
            background-color: #3A3A3A !important;
        }

        .status-note {
            text-align: center;
            color: #9A9A9A;
            font-size: 0.85rem;
            margin-top: 2.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-title">Web App for Exploratory Data Analysis and ML Workflows</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">A dataset-agnostic workspace for exploring, preparing, and modeling your data — built for transparency at every step.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-tags">Structured Data · Unstructured Data | Explore · Preprocess · Model · Inspect</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="hero-divider">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Two entry-point cards
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="path-card">
            <div class="path-icon icon-structured">📊</div>
            <div class="path-title">Structured Data</div>
            <div class="path-desc">
                Work with tabular datasets — upload CSV, Excel, or JSON files and move through
                preprocessing, EDA, dimensionality reduction, regression, classification, and clustering.
            </div>
            <div class="path-chip-row">
                <span class="path-chip chip-blue">CSV</span>
                <span class="path-chip chip-blue">Excel</span>
                <span class="path-chip chip-blue">JSON</span>
                <span class="path-chip chip-blue">ML Models</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.link_button("Go to ML & EDA Dashboard →", "http://localhost:8501", use_container_width=True)

with col2:
    st.markdown(
        """
        <div class="path-card">
            <div class="path-icon icon-unstructured">📝</div>
            <div class="path-title">Unstructured Data</div>
            <div class="path-desc">
                Work with text — upload TXT, LOG, PDF, DOCX, or PPTX files and move through
                cleaning, lexical analysis, vectorization, text classification, and topic modeling.
            </div>
            <div class="path-chip-row">
                <span class="path-chip chip-pink">Text</span>
                <span class="path-chip chip-pink">PDF / DOCX / PPTX</span>
                <span class="path-chip chip-pink">NLP</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.link_button("Go to NLP Dashboard →", "http://localhost:8502", use_container_width=True)

st.markdown(
    '<div class="status-note">More unstructured data types — images, audio — are on the roadmap.</div>',
    unsafe_allow_html=True,
)
