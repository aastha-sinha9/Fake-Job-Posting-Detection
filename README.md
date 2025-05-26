# Fake Job Posting Detection

This project aims to identify fake job postings using a machine learning model trained on textual data. It includes both a backend ML model and a deployed frontend interface for user interaction.

##  Problem Statement

Fake job postings have increased across online platforms, misleading job seekers and wasting time. This project attempts to classify job descriptions as **genuine** or **fake** using natural language processing and deep learning techniques.

---

##  Features

- Preprocessed job posting dataset
- Word embedding using GloVe vectors
- Deep learning model (ANN) for binary classification
- Interactive frontend built using [Bolt AI](https://bolt.ai)
- Fully deployed frontend on [Netlify](https://www.netlify.com/)
- API-ready backend (Flask or Streamlit compatible)

---

##  Machine Learning Workflow

1. **Data Cleaning**
   - Removed nulls, unnecessary columns, and duplicate rows
   - Balanced the dataset if needed

2. **Text Preprocessing**
   - Tokenization, stop word removal, lowercasing
   - Embedding with `glove.6B.100d.txt` vectors

3. **Model**
   - Deep Neural Network with dense layers
   - Accuracy, precision, recall as evaluation metrics

4. **Deployment**
   - Streamlit or Flask backend (as per your final setup)
   - Frontend hosted on Netlify

---

## 🌐 Frontend Deployment

The frontend was built using **Bolt AI** and deployed via **Netlify**.

🔗 Live Website: [https://your-netlify-url.netlify.app](https://your-netlify-url.netlify.app)

You can enter a job description and get real-time predictions from the backend.

---

## 📁 Project Structure
Fake-Job-Posting-Detection/
│
├── Glove/ # Pre-trained word embeddings (not pushed to GitHub)
├── frontend/ # Bolt AI frontend files (optional if linked externally)
├── model/ # Model training and evaluation notebooks
├── app.py # Backend script (Flask or Streamlit)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore
Dataset
Dataset used: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

~18,000 job postings

Balanced for binary classification


---

##  Setup Instructions

### Clone the Repo

```bash
git clone https://github.com/yourusername/fake-job-posting-detection.git
cd fake-job-posting-detection

pip install -r requirements.txt

streamlit run app.py

Job Description:
"Looking for an experienced data scientist to work on a variety of analytics projects..."
```


Author
Aastha
Engineering Student at NSUT
