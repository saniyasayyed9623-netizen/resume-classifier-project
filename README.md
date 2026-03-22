# 🧠 Resume Classifier AI 

# Overview
Resume Classifier AI automatically classifies resumes into job categories using a trained SVM + TF-IDF model. It extracts candidate details, scores resumes by category-specific criteria, and provides visual analytics.

## 🚀 Live Demo:
<img width="1394" height="888" alt="image" src="https://github.com/user-attachments/assets/b85fbcb5-2862-4197-88de-94b7ab3a5043" />

( https://resume-classifier-project-fvt4grsxilbm2tnckszizb.streamlit.app/)

## Features
Multi-Format Support: Extracts text from both .pdf and .docx files.

Advanced Preprocessing: Uses NLTK and Regular Expressions for text cleaning.

Smart Classification: Built using a Support Vector Machine (SVM) model.

Interactive Dashboard: Real-time resume analysis using Streamlit.

## 📂Project Structure
NLP_Resumes_Classification_commented.ipynb: Detailed Jupyter Notebook with logic and Q&A.

- app.py: Streamlit application script.

- Data/: Folder containing the dataset of 79 resumes.

- models/: Pre-trained model components (.pkl files).

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Scikit-learn, NLTK, spaCy, PyPDF2, python-docx
- Model Serialization: Joblib
- **Deployment:** Streamlit Cloud / Local

## 📊 How it Works
1. **Upload:** User uploads a resume (PDF or Word).
2. **Process:** The system cleans the text and extracts key entities.
3. **Predict:** The trained SVM model predicts the most suitable job category.
4. **Result:** Dashboard shows the category, extracted name/email, and skill charts.

_ _ _
*Note: This project was developed as part of a Data Science internship/academic project.*
