import streamlit as st
import pickle
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained models and vectorizer
clf = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

# Define a function to clean text data
def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Define skill extraction function
def extract_skills(text):
    skills = ["Python", "SQL", "Machine Learning", "NLP", "Data Analysis", "Java", "Excel"]
    doc = nlp(text)
    extracted_skills = [token.text for token in doc if token.text in skills]
    return extracted_skills

# Define main function for the app
def main():
    st.title("Automated Resume Screening Tool")
    st.write("Upload a resume, and the tool will predict the job category.")

    # Upload file
    uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf"])
    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8")
        cleaned_text = clean_resume(resume_text)
        extracted_skills = extract_skills(cleaned_text)
        input_vector = tfidf.transform([cleaned_text])
        prediction = clf.predict(input_vector)[0]

        # Display the result
        st.write("**Predicted Job Category:**", prediction)
        st.write("**Extracted Skills:**", extracted_skills)

# Run the main function
if __name__ == "__main__":
    main()
