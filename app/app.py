import streamlit as st
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Resume Classifier")
resume_text = st.text_area("Paste Resume Text")

if st.button("Classify"):
    vector = tfidf.transform([resume_text])
    pred = model.predict(vector)
    st.success(f"Predicted Job Role: {pred[0]}")
