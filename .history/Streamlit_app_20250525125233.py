import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model('ann_model.h5')

# Constants
MAX_LENGTH = 100

# Title
st.title("Fraudulent Job Post Detector")

# Input
user_input = st.text_area("Paste the job description here")

# Prediction
if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        st.error("🚨 This job post seems fraudulent!")
    else:
        st.success("✅ This job post looks legit.")
