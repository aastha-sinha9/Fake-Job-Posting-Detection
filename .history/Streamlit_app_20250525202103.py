import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


import gdown
import os

# Google Drive links (make sure they are direct file links from Google Drive Direct Link Generator)
MODEL_URL = "https://drive.google.com/file/d/18gMtzQ-lwn-BO580P9qjf27b1PPC7oFN/view?usp=sharing"
TOKENIZER_URL = ""

# Download files if they don't exist
if not os.path.exists("ann_model.h5"):
    gdown.download(MODEL_URL, "ann_model.h5", quiet=False)

if not os.path.exists("tokenizer.pkl"):
    gdown.download(TOKENIZER_URL, "tokenizer.pkl", quiet=False)

# Load model and tokenizer
model = tf.keras.models.load_model("ann_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


MAX_LEN = 200 


st.set_page_config(page_title="Fake Job Detection", layout="centered")
st.title("🕵️ Fraud Job Posting Detector")
st.markdown("""
Enter a job description and we'll help detect if it's likely **fraudulent** or **genuine** using our trained neural network. 🚨
""")


job_text = st.text_area("Paste the job description below:", height=250)

if st.button("Predict"):
    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    else:
        
        seq = tokenizer.texts_to_sequences([job_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        
        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            st.error(f"🚫 This job posting looks **fraudulent** with confidence {pred*100:.2f}%")
        else:
            st.success(f"✅ This job posting looks **genuine** with confidence {(1 - pred)*100:.2f}%")


st.markdown("---")
st.markdown("Built with ❤️ by Aastha using Deep Learning and GloVe embeddings.")
