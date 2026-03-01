import streamlit as st
import pandas as pd
import os
import docx2txt
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="CareerBridge AI", layout="wide")

def extract_text(file):
    if file.type == "application/pdf":
        return " ".join([page.extract_text() for page in PdfReader(file).pages])
    return docx2txt.process(file)

# --- 2 & 3. Data Handling (JobsDB & LinkedIn) ---
def get_mock_data(type="jobs"):
    if type == "jobs":
        return [{"title": f"Software Engineer {i}", "company": "Tech Corp", "location": "Remote", "summary": "Full stack role", "link": "https://jobsdb.com", "text": "Python React AWS"} for i in range(30)]
    return [{"name": f"Mentor {i}", "title": "Senior Dev", "summary": "Ex-Google engineer", "link": "https://linkedin.com", "text": "Mentorship scaling systems"} for i in range(30)]

def load_data(filename, data_type):
    if not os.path.exists(filename):
        # Placeholder for RapidAPI logic; using mock data for now
        df = pd.DataFrame(get_mock_data(data_type))
        df.to_csv(filename, index=False)
    return pd.read_csv(filename)

jobs_df = load_data("jobsdb.csv", "jobs")
linkedin_df = load_data("linkedin.csv", "mentors")

# --- 4. LLM Functions (Placeholders) ---
def analyze_cv(text):
    # Integration point for OpenAI/Anthropic
    return "Your profile shows strong backend expertise.", "Add more quantifiable achievements in your latest role."

# --- 5 & 6. Semantic Search Logic ---
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_matches(query, target_df, text_col):
    model = get_model()
    query_emb = model.encode(query, convert_to_tensor=True)
    target_embs = model.encode(target_df[text_col].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(query_emb, target_embs)[0]
    target_df['score'] = scores.tolist()
    return target_df.sort_values(by='score', ascending=False)

# --- 7. Layout ---
st.title("🚀 CareerBridge AI")
st.subheader("Bridge the gap to your dream career")

# CV Upload Section
st.info("Upload your resume and let AI find your perfect job matches on JobsDB and connect you with mentors who've walked your path.")
uploaded_file = st.file_uploader("Choose a CV (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    cv_text = extract_text(uploaded_file)
    analysis, suggestions = analyze_cv(cv_text)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Analysis")
        st.write(analysis)
    with col2:
        st.markdown("### Suggestions to improve CV")
        st.write(suggestions)

    st.divider()

    # Lower Section: Matching
    left_col, right_col = st.columns(2)

    with left_col:
        st.header("🎯 Matched Jobs")
        matched_jobs = get_matches(cv_text, jobs_df, 'text')
        
        # Simple Pagination
        page_j = st.number_input("Jobs Page", min_value=1, value=1)
        start_j = (page_j - 1) * 10
        for _, job in matched_jobs.iloc[start_j : start_j+10].iterrows():
            with st.expander(f"{job['title']} - {job['company']} (Score: {job['score']:.2f})"):
                st.write(f"**Location:** {job['location']}")
                st.write(job['summary'])
                st.link_button("View on JobsDB", job['link'])

    with right_col:
        st.header("🤝 Career Path Mentors")
        matched_mentors = get_matches(cv_text, linkedin_df, 'text')
        
        page_m = st.number_input("Mentors Page", min_value=1, value=1)
        start_m = (page_m - 1) * 10
        for _, mentor in matched_mentors.iloc[start_m : start_m+10].iterrows():
            with st.container(border=True):
                st.write(f"**{mentor['name']}** - {mentor['title']}")
                st.write(mentor['summary'])
                st.link_button("LinkedIn Profile", mentor['link'])
                if st.button(f"Coffee chat Invite", key=f"btn_{mentor['name']}"):
                    st.code(f"Hi {mentor['name']}, I saw your profile and love your work at {mentor['title']}. Would love to grab a virtual coffee!")
