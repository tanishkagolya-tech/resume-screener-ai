import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ---- CONFIG ----
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

model = SentenceTransformer('all-MiniLM-L6-v2')

KEYWORDS = [
    "python", "sql", "pandas", "numpy", "matplotlib", "seaborn",
    "scikit-learn", "sklearn", "pytorch", "tensorflow", "keras",
    "machine learning", "deep learning", "nlp", "data analysis",
    "data cleaning", "feature engineering", "power bi", "excel",
    "tableau", "statistics", "probability", "regression", "classification",
    "clustering", "neural network", "computer vision", "api",
    "flask", "fastapi", "streamlit", "git", "github", "docker",
    "spark", "hadoop", "mongodb", "mysql", "postgresql",
    "data visualization", "etl", "pipeline", "aws", "gcp", "azure",
    "java", "c++", "javascript", "react", "html", "css"
]

def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text.lower()

def get_match_score(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]
    score = float(np.clip(score, 0, 1))
    return round(score * 100, 1)

def get_keywords(resume_text, jd_text):
    matched = []
    missing = []
    for keyword in KEYWORDS:
        in_jd = keyword in jd_text.lower()
        in_resume = keyword in resume_text.lower()
        if in_jd and in_resume:
            matched.append(keyword)
        elif in_jd and not in_resume:
            missing.append(keyword)
    return matched, missing

def get_ai_suggestions(resume_text, jd_text, missing_keywords, score):
    prompt = f"""
You are a professional resume coach helping a 3rd year AI & Data Science student improve their resume.

Job Description:
{jd_text}

Resume Content:
{resume_text[:2000]}

Match Score: {score}%
Missing Keywords: {', '.join(missing_keywords) if missing_keywords else 'None'}

Give exactly 4 short, specific, actionable suggestions to improve this resume for this job.
Each suggestion should be on a new line starting with a number.
Be encouraging and beginner-friendly. Keep each suggestion under 2 sentences.
"""
    response = gemini.generate_content(prompt)
    return response.text

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Resume Screener AI",
    page_icon="🎯",
    layout="centered"
)

# ---- CSS ----
st.markdown("""
<style>
    .stApp {
        background-color: #f1f5f9;
    }
    .block-container {
        padding: 2.5rem 3rem;
        max-width: 860px;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 12px;
        font-size: 14px;
        color: #1e293b;
        padding: 12px;
        transition: border 0.2s;
    }
    .stTextArea textarea:focus {
        border: 1.5px solid #185FA5;
        box-shadow: 0 0 0 3px rgba(24,95,165,0.1);
    }
    section[data-testid="stFileUploadDropzone"] {
        background-color: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        transition: border 0.2s;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #185FA5;
    }
    .stButton > button {
        background-color: #185FA5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.5rem;
        font-size: 15px;
        font-weight: 600;
        width: auto;
        letter-spacing: 0.3px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #1a4f8a;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(24,95,165,0.3);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }
    .stSuccess {
        background-color: #f0fdf4 !important;
        border: 1px solid #86efac !important;
        border-radius: 10px !important;
        color: #166534 !important;
        font-weight: 500;
    }
    .stError {
        background-color: #fff1f2 !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 10px !important;
        color: #9f1239 !important;
        font-weight: 500;
    }
    .stWarning {
        background-color: #fffbeb !important;
        border: 1px solid #fcd34d !important;
        border-radius: 10px !important;
        color: #92400e !important;
        font-weight: 500;
    }
    hr {
        border: none;
        border-top: 1.5px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    div[data-testid="stHorizontalBlock"] > div {
        background: white;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    label, .stTextArea label, .stFileUploader label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="
        font-size: 2.6rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -1.2px;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    ">🎯 Resume Screener AI</h1>
    <p style="
        color: #64748b;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    ">Upload your resume · Paste any JD · Get AI-powered insights instantly</p>
</div>
<hr>
""", unsafe_allow_html=True)

# ---- INPUTS ----
jd_text = st.text_area("📋 Paste Job Description here", height=200)
uploaded_file = st.file_uploader("📄 Upload Your Resume (PDF)", type="pdf")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Analyze My Resume"):
    if uploaded_file and jd_text:

        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            score = get_match_score(resume_text, jd_text)
            matched, missing = get_keywords(resume_text, jd_text)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- SCORE CARD ----
        if score >= 70:
            score_color = "#16a34a"
            bar_color = "#16a34a"
        elif score >= 50:
            score_color = "#d97706"
            bar_color = "#d97706"
        else:
            score_color = "#dc2626"
            bar_color = "#dc2626"

        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.75rem 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        ">
            <p style="
                font-size: 12px;
                font-weight: 700;
                color: #94a3b8;
                letter-spacing: 1.5px;
                margin-bottom: 8px;
                text-transform: uppercase;
            ">Match Score</p>
            <h1 style="
                font-size: 3.5rem;
                font-weight: 800;
                color: {score_color};
                margin: 0;
                line-height: 1;
                letter-spacing: -2px;
            ">{score}%</h1>
            <div style="
                background: #f1f5f9;
                height: 10px;
                border-radius: 99px;
                margin-top: 1.25rem;
                overflow: hidden;
            ">
                <div style="
                    background: {bar_color};
                    width: {score}%;
                    height: 10px;
                    border-radius: 99px;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if score >= 70:
            st.success("✅ Great match! Your resume fits this job well.")
        elif score >= 50:
            st.warning("⚠️ Decent match. Consider adding more relevant keywords.")
        else:
            st.error("❌ Low match. Your resume needs more alignment with the JD.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- KEYWORDS ----
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matched Keywords")
            if matched:
                for kw in matched:
                    st.success(f"✓ {kw}")
            else:
                st.info("No matching keywords found.")

        with col2:
            st.markdown("### ❌ Missing Keywords")
            if missing:
                for kw in missing:
                    st.error(f"✗ {kw}")
            else:
                st.success("🎉 No missing keywords! Great job.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- AI SUGGESTIONS ----
        st.markdown("### 🤖 AI-Powered Suggestions")
        with st.spinner("Generating personalized suggestions with Gemini..."):
            suggestions = get_ai_suggestions(resume_text, jd_text, missing, score)

        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #bae6fd;
            border-left: 4px solid #185FA5;
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            font-size: 14.5px;
            line-height: 1.9;
            color: #1e293b;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        ">{suggestions.replace(chr(10), '<br>')}</div>
        """, unsafe_allow_html=True)

    else:
        st.error("⚠️ Please upload a resume AND paste the job description first.")

# ---- FOOTER ----
st.markdown("""
<br><br>
<hr>
<p style="
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.3px;
">
    Built with Python · Streamlit · Sentence Transformers · Google Gemini API
</p>
""", unsafe_allow_html=True)