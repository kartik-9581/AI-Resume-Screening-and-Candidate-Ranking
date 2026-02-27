import streamlit as st
import os
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text


# ----------------------------
# Extract Text from PDF
# ----------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Resume Screening App", layout="wide")

st.title("🧠 Resume Screening & Candidate Ranking System")
st.write("Upload resumes and enter job description to rank candidates automatically.")

# Job Description Input
job_description = st.text_area("📄 Enter Job Description")

# Upload Multiple Resumes
uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🔍 Rank Candidates"):

    if not job_description or not uploaded_files:
        st.warning("Please upload resumes and enter a job description.")
    else:
        resume_texts = []
        resume_names = []

        # Extract and clean resume text
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            cleaned = clean_text(text)
            resume_texts.append(cleaned)
            resume_names.append(file.name)

        # Clean job description
        cleaned_jd = clean_text(job_description)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([cleaned_jd] + resume_texts)

        # Compute Cosine Similarity
        similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Create Results DataFrame
        results = pd.DataFrame({
            "Candidate Name": resume_names,
            "Matching Score (%)": similarity_scores * 100
        })

        # Sort by Highest Score
        results = results.sort_values(by="Matching Score (%)", ascending=False)

        st.success("✅ Ranking Completed!")
        st.dataframe(results.reset_index(drop=True))

        # Highlight Top Candidate
        st.subheader("🏆 Top Candidate")
        st.write(results.iloc[0])
