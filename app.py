import streamlit as st
import pandas as pd
import os
import requests
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# --- Configuration & API Keys ---
# Set your OpenAI API key or use Streamlit Secrets for production
OPENAI_API_KEY = "your_openai_api_key"
RAPID_API_KEY = "your_rapidapi_key"

# --- Data Preparation (Requirements 2 & 3) ---
def get_data(filename, api_url, api_host):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        # Placeholder for RapidAPI call logic
        # headers = {"X-RapidAPI-Key": RAPID_API_KEY, "X-RapidAPI-Host": api_host}
        # response = requests.get(api_url, headers=headers)
        # data = response.json()
        
        # Creating sample data if API is not configured
        if "jobsdb" in filename:
            df = pd.DataFrame([
                {"title": f"Software Engineer {i}", "company": "Tech Corp", "location": "Hong Kong", "summary": "Full stack role...", "link": "https://hk.jobsdb.com", "text": "Python React SQL"} for i in range(50)
            ])
        else:
            df = pd.DataFrame([
                {"name": f"Mentor {i}", "title": "Senior Dev", "summary": "10 years exp...", "link": "https://linkedin.com", "text": "Cloud AI Mentorship"} for i in range(50)
            ])
        df.to_csv(filename, index=False)
        return df

# --- Utility Functions ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    return ""

def call_llm(prompt):
    # Simplified LLM call using OpenAI
    try:
        # client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # response = client.chat.completions.create(...)
        return f"LLM Summary/Suggestion based on prompt: {prompt[:50]}..."
    except:
        return "LLM service unavailable. Check API key."

def semantic_match(query_text, dataset_df):
    vectorizer = TfidfVectorizer()
    all_texts = [query_text] + dataset_df['text'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    dataset_df['match_score'] = (scores * 100).round(2)
    return dataset_df.sort_values(by='match_score', ascending=False)

def paginate(df, page_size=10):
    total_pages = (len(df) // page_size) + (1 if len(df) % page_size > 0 else 0)
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
    start_idx = (page_num - 1) * page_size
    return df.iloc[start_idx : start_idx + page_size]

# --- Page Definitions (Requirements 4, 5, 6) ---
def cv_upload_page():
    st.title("📄 CV Upload & Analysis")
    uploaded_file = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        cv_text = extract_text(uploaded_file)
        st.session_state['cv_text'] = cv_text
        st.success("CV uploaded successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary")
            st.write(call_llm(f"Summarize this CV: {cv_text[:1000]}"))
        with col2:
            st.subheader("Improvement Suggestions")
            st.write(call_llm(f"Suggest improvements for: {cv_text[:1000]}"))

def job_match_page():
    st.title("💼 JobsDB Matching")
    if 'cv_text' not in st.session_state:
        st.warning("Please upload a CV first.")
        return
    
    jobs_df = get_data("jobsdb.csv", "API_URL", "jdb.p.rapidapi.com")
    matched_jobs = semantic_match(st.session_state['cv_text'], jobs_df)
    
    paged_jobs = paginate(matched_jobs)
    for _, job in paged_jobs.iterrows():
        with st.expander(f"{job['title']} @ {job['company']} (Score: {job['match_score']}%)"):
            st.write(f"**Location:** {job['location']}")
            st.write(f"**Summary:** {job['summary']}")
            st.link_button("View on JobsDB", job['link'])

def mentor_match_page():
    st.title("🤝 Career Path Mentors")
    if 'cv_text' not in st.session_state:
        st.warning("Please upload a CV first.")
        return
    
    mentor_df = get_data("linkedin.csv", "API_URL", "li.p.rapidapi.com")
    matched_mentors = semantic_match(st.session_state['cv_text'], mentor_df)
    
    paged_mentors = paginate(matched_mentors)
    for _, mentor in paged_mentors.iterrows():
        st.markdown(f"### {mentor['name']} - {mentor['title']}")
        st.write(mentor['summary'])
        st.link_button("LinkedIn Profile", mentor['link'])
        
        if st.button(f"Coffee chat Invite", key=f"btn_{mentor['name']}"):
            st.info(f"Greeting: Hi {mentor['name']}, I saw your profile and would love to chat about your career path!")

# --- Main Navigation Setup ---
pg = st.navigation([
    st.Page(cv_upload_page, title="CV Upload", icon="📤"),
    st.Page(job_match_page, title="Job Matching", icon="🔍"),
    st.Page(mentor_match_page, title="Mentors", icon="👥")
])
pg.run()
