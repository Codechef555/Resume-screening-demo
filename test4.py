import streamlit as st
import pdfplumber
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    # Remove unwanted characters and extra white spaces
    text = " ".join(text.split())
    return text

# Load the spacy model for English language
nlp = spacy.load("en_core_web_sm")

# Create a TF-IDF vectorizer with cosine similarity
vectorizer = TfidfVectorizer(stop_words="english")

# Create the web interface using Streamlit
st.title("Resume shortlisting")

# Input field for username
username = st.text_input("Enter your username:")

# Input fields for job description and PDF files
st.subheader("Job Description")
job_description = st.text_area("Enter the job description for a software engineer:")
pdf_files = st.file_uploader("Upload up to 10 PDF files", accept_multiple_files=True)

# List to store qualified and unqualified candidates
qualified_candidates = []
unqualified_candidates = []

# If PDF files and job description are uploaded, match them and display the results
if pdf_files and job_description:
    for pdf_file in pdf_files:
        # Convert PDF to clean text
        clean_text = extract_text_from_pdf(pdf_file)
        # Create a spaCy doc object for the job description
        job_description_doc = nlp(job_description)
        # Create a spaCy doc object for the clean text
        clean_text_doc = nlp(clean_text)
        # Get the TF-IDF vectors for the job description and the clean text
        job_description_vector = vectorizer.fit_transform([job_description])
        clean_text_vector = vectorizer.transform([clean_text])
        # Calculate the cosine similarity between the job description and the clean text vectors
        similarity = cosine_similarity(job_description_vector, clean_text_vector)
        # Determine if the resume is qualified or not based on the similarity score
        if similarity[0][0] >= 0.7:
            qualified_candidates.append(pdf_file.name)
        else:
            unqualified_candidates.append(pdf_file.name)

    # Print the list of qualified and unqualified candidates
    st.subheader("Qualified Candidates:")
    for candidate in qualified_candidates:
        st.write(candidate)
    st.subheader("Unqualified Candidates:")
    for candidate in unqualified_candidates:
        st.write(candidate)

    # Show the number of qualified and unqualified candidates in a line chart
    chart_data = pd.DataFrame({"Qualified": [len(qualified_candidates)], "Unqualified": [len(unqualified_candidates)]})
    st.line_chart(chart_data)

    # Print the selected candidates with the username
    st.subheader("Selected Candidates:")
    if len(qualified_candidates) > 0:
        st.write(f"{username}, you have selected the following {len(qualified_candidates)} candidates for the interview:")
        for candidate in qualified_candidates:
            st.write(candidate)
    else:
        st.write(f"Sorry {username}, no candidates were selected for the interview.")
