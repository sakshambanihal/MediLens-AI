
import streamlit as st
import joblib
import re
import html
import os
from collections import Counter
from io import BytesIO
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer
)

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="MediLens AI - Medical Report Analyzer",
    page_icon=None,
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    saved = joblib.load("medical_report_svm.pkl")
    return saved["tfidf"], saved["model"]

try:
    tfidf, model = load_model()
except Exception:
    st.error(
        "Model file not found. Make sure "
        "medical_report_svm.pkl is in the project folder."
    )
    st.stop()

# ---------------- THEME ----------------
if "light_mode" not in st.session_state:
    st.session_state.light_mode = True

if st.session_state.light_mode:
    bg = "#F2F7FD"
    panel = "#FFFFFF"
    text = "#203047"
    muted = "#64748B"
    accent = "#3978C5"
    soft = "#E5F0FF"
    border = "#D5E4F5"
else:
    bg = "#111827"
    panel = "#1B2638"
    text = "#E5EDF8"
    muted = "#A5B4C8"
    accent = "#76A9FA"
    soft = "#253750"
    border = "#34445B"

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
}}

.block-container {{
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

h1, h2, h3, p, label {{
    color: {text};
}}

.header-card {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 24px;
    margin: 14px 0 22px 0;
}}

.subtitle {{
    color: {muted};
    line-height: 1.7;
}}

.result-card {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}}

.result-label {{
    color: {muted};
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.result-value {{
    color: {accent};
    font-size: 23px;
    font-weight: 700;
    margin-top: 10px;
}}

.keyword {{
    display: inline-block;
    background: {soft};
    color: {text};
    border: 1px solid {border};
    border-radius: 5px;
    padding: 4px 8px;
    margin: 3px;
    font-size: 13px;
}}

.report-text {{
    background: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 18px;
    line-height: 1.9;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}

.report-text mark {{
    background: #FFE58F;
    color: #332900;
    padding: 2px;
    border-radius: 3px;
}}

.stButton > button,
.stDownloadButton > button {{
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
}}

hr {{
    border-color: {border};
}}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
theme_col, empty_col = st.columns([1, 5])

with theme_col:
    theme_label = (
        "Switch to dark mode"
        if st.session_state.light_mode
        else "Switch to light mode"
    )
    if st.button(theme_label):
        st.session_state.light_mode = not st.session_state.light_mode
        st.rerun()

# Optional custom logo: save it as logo.png
if os.path.exists("logo.png"):
    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        st.image("logo.png", width=90)
    with title_col:
        st.title("MediLens AI")
        st.caption("AI-Based Medical Report Analyzer")
else:
    st.title("MediLens AI")
    st.caption("AI-Based Medical Report Analyzer")

st.markdown("""
<div class="header-card">
    <h3>Medical Report Analysis</h3>
    <div class="subtitle">
        Extract important terms, read a short summary and
        download the analysis for future reference.
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "For educational and informational use only. "
    "This tool does not provide medical diagnosis or treatment."
)

# ---------------- INPUT ----------------
st.subheader("Upload Medical Report")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)

pasted_text = st.text_area(
    "Or paste report text here",
    height=170,
    placeholder="Paste the medical report text here..."
)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    if file.name.lower().endswith(".txt"):
        return file.getvalue().decode(
            "utf-8-sig", errors="replace"
        )

    reader = PdfReader(file)
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

# ---------------- KEYWORDS ----------------
STOP_WORDS = {
    "with", "from", "that", "this", "were", "have",
    "been", "patient", "there", "their", "which",
    "into", "report", "after", "before", "about",
    "under", "over", "during", "also", "then",
    "than", "they", "them", "these", "those",
    "when", "where", "what", "while", "will",
    "would", "could", "should", "such", "each",
    "some", "very", "more", "most", "only",
    "other", "history", "clinical", "present",
    "noted", "showed", "found", "performed",
    "following", "without", "because", "medical",
    "examination", "recommended"
}

def extract_keywords(text, limit=12):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    counts = Counter(
        word for word in words
        if word not in STOP_WORDS
    )

    return [word for word, count in counts.most_common(limit)]

# ---------------- SMART SUMMARY ----------------
def make_summary(text, max_sentences=3):
    # Basic extractive summary: selects sentences from the
    # original text instead of generating new medical claims.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [
        s.strip() for s in sentences
        if len(s.split()) >= 5
    ]

    if not sentences:
        return text.strip()

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    frequencies = Counter(
        w for w in words if w not in STOP_WORDS
    )

    scored = []
    for index, sentence in enumerate(sentences):
        terms = re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower())
        score = sum(frequencies[w] for w in terms)
        scored.append((score, index, sentence))

    selected = sorted(
        sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences],
        key=lambda x: x[1]
    )

    return " ".join(item[2] for item in selected)

# ---------------- HIGHLIGHT TEXT ----------------
def highlight_text(text, keywords):
    escaped = html.escape(text)

    if not keywords:
        return escaped.replace("\n", "<br>")

    pattern = re.compile(
        r"\b(" + "|".join(
            re.escape(k) for k in sorted(keywords, key=len, reverse=True)
        ) + r")\b",
        flags=re.IGNORECASE
    )

    highlighted = pattern.sub(
        lambda match: "<mark>" + match.group(0) + "</mark>",
        escaped
    )

    return highlighted.replace("\n", "<br>")

# ---------------- PDF GENERATION ----------------
def create_pdf(prediction, summary, keywords, report_text):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#285A91"),
        alignment=TA_LEFT,
        fontSize=20,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#285A91"),
        spaceBefore=12,
        spaceAfter=7
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=7,
        wordWrap="CJK"
    )

    story = [
        Paragraph("MediLens AI", title_style),
        Paragraph("Medical Report Analysis", heading_style),
        Paragraph(
            "<b>Predicted specialty:</b> " + html.escape(prediction),
            body_style
        ),
        Paragraph("Smart Summary", heading_style),
        Paragraph(html.escape(summary), body_style),
        Paragraph("Important Keywords", heading_style),
        Paragraph(
            html.escape(", ".join(keywords) if keywords else "None found"),
            body_style
        ),
        Paragraph("Extracted Report Text", heading_style),
        Paragraph(
            html.escape(report_text).replace("\n", "<br/>"),
            body_style
        ),
        Spacer(1, 15),
        Paragraph(
            "Educational and informational use only. "
            "This document does not provide medical diagnosis or treatment.",
            body_style
        )
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- ANALYZE ----------------
if st.button("Analyze Report", use_container_width=True):
    try:
        if uploaded_file is not None:
            report_text = extract_text(uploaded_file)
        else:
            report_text = pasted_text

        if not report_text.strip():
            st.error("Please upload a readable report or paste text.")
        else:
            vector = tfidf.transform([report_text])
            prediction = str(model.predict(vector)[0])
            keywords = extract_keywords(report_text)
            summary = make_summary(report_text)

            st.session_state.analysis = {
                "text": report_text,
                "prediction": prediction,
                "keywords": keywords,
                "summary": summary
            }

    except Exception as e:
        st.error(f"Could not analyze the report: {e}")

# ---------------- DISPLAY RESULTS ----------------
if "analysis" in st.session_state:
    result = st.session_state.analysis

    report_text = result["text"]
    prediction = result["prediction"]
    keywords = result["keywords"]
    summary = result["summary"]

    st.divider()
    st.header("Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Specialty</div>
            <div class="result-value">{html.escape(prediction)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Report Overview</div>
            <div class="result-value">{len(report_text.split())} words</div>
        </div>
        """, unsafe_allow_html=True)

    # Smart Summary
    st.subheader("Smart Summary")
    st.markdown(
        "<div class='result-card'>" +
        html.escape(summary) +
        "</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "This is a basic extractive summary made from the original text. "
        "It is not a clinical interpretation."
    )

    # Keywords
    st.subheader("Important Keywords")

    if keywords:
        chips = "".join(
            f"<span class='keyword'>{html.escape(word)}</span>"
            for word in keywords
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.write("No keywords found.")

    # Highlighted report
    st.subheader("Report with Highlighted Keywords")
    highlighted = highlight_text(report_text, keywords)

    st.markdown(
        f"<div class='report-text'>{highlighted}</div>",
        unsafe_allow_html=True
    )

    # Download analysis PDF
    pdf_data = create_pdf(
        prediction,
        summary,
        keywords,
        report_text
    )

    st.download_button(
        "Download Analysis as PDF",
        data=pdf_data,
        file_name="medical_report_analysis.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.divider()
st.caption(
    "MediLens AI | NLP and Machine Learning | Educational use only"
)
