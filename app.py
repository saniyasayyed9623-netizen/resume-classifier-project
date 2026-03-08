import streamlit as st
import joblib
import re
import os
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from docx import Document
import PyPDF2
from datetime import datetime

# --- Stopwords Setup ---
# Downloading NLTK stopwords for text cleaning
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- spaCy NLP Model Load ---
# Using spaCy for smart Name, Email, Phone entity extraction
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# --- Page Settings ---
st.set_page_config(page_title="Group 4 | Resume AI", page_icon="🧠", layout="wide")

# --- Theme is applied via .streamlit/config.toml (no HTML/CSS needed) ---


# --- Session State for Upload History ---
# Initializing session state to store upload history across reruns
if "history" not in st.session_state:
    st.session_state.history = []

# --- Model Loading Logic ---
@st.cache_resource
def load_models():
    # Loading the SVM model, TF-IDF vectorizer, and label encoder from models folder
    try:
        model = joblib.load('models/svm_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        le    = joblib.load('models/label_encoder.pkl')
        return model, tfidf, le
    except Exception as e:
        st.error(f"Model files not found: {e}")
        return None, None, None

model, tfidf, le = load_models()

# --- File Extraction Function ---
def get_text_from_file(file):
    # Extracting text from PDF, DOCX, or TXT formats
    if file.name.endswith('.docx'):
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    elif file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return file.read().decode('utf-8', errors='ignore')

# --- Text Cleaning Function ---
def clean_resume_text(text):
    # Cleaning raw text by removing URLs, special characters, and stopwords
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Prediction Function ---
def predict_category(cleaned_text):
    # Vectorizing cleaned text and predicting the resume category
    vectorized = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized)
    category = le.inverse_transform(prediction)[0]
    # Remapping General_Developer label to React Developer for display
    if category == "General_Developer":
        category = "React Developer"
    return category

# --- Resume Detail Extraction Functions ---

def extract_name(text):
    # Step 1: Check for explicit "Name:" label in resume
    label_match = re.search(r'(?:Name\s*[:\-]\s*)([A-Za-z]+(?: [A-Za-z]+){1,3})', text, re.IGNORECASE)
    if label_match:
        return label_match.group(1).strip().title()

    # Step 2: Use spaCy NER to find PERSON entity automatically
    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if 2 <= len(name.split()) <= 4:
                return name.title()

    # Step 3: Scan first 10 lines for a name-like clean line
    skip_keywords = [
        'resume', 'curriculum', 'vitae', 'cv', 'objective', 'summary',
        'experience', 'education', 'skill', 'project', 'contact',
        'address', 'phone', 'email', 'linkedin', 'github', 'profile',
        'declaration', 'reference', 'achievement', 'certification',
        'seeking', 'looking', 'developer', 'engineer', 'manager'
    ]
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:10]:
        words = line.split()
        if 2 <= len(words) <= 4 and len(line) < 60:
            if not re.search(r'[@:0-9\|\\/,\.\(\)]', line):
                if not any(kw in line.lower() for kw in skip_keywords):
                    if all(w[0].isupper() for w in words if w.isalpha()):
                        return line.title()

    return "Not Found"

def extract_email(text):
    # Step 1: Check for "Email:", "E-mail:", "Mail:" label
    label_match = re.search(
        r'(?:e[\-]?mail|mail|contact)\s*[:\-]\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
        text, re.IGNORECASE
    )
    if label_match:
        return label_match.group(1).strip()

    # Step 2: spaCy se email find karo (EMAIL entity)
    doc = nlp(text[:2000])
    for token in doc:
        if token.like_email:
            return token.text.strip()

    # Step 3: Regex fallback — any email pattern in full text
    match = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    valid = [m for m in match if not re.search(r'\.(png|jpg|jpeg|gif|svg|pdf|docx)$', m, re.IGNORECASE)]
    return valid[0] if valid else "Not Found"

def extract_phone(text):
    # Step 1: Check for "Phone:", "Mobile:", "Contact:", "Cell:", "Tel:" label
    label_match = re.search(
        r'(?:phone|mobile|contact|cell|ph|tel|mob)\s*[:\-]?\s*([\+\(]?[0-9][0-9\s\-\(\)]{8,}[0-9])',
        text, re.IGNORECASE
    )
    if label_match:
        digits = re.sub(r'\D', '', label_match.group(1))
        if 10 <= len(digits) <= 15:
            return label_match.group(1).strip()

    # Step 2: Indian mobile number — 10 digits starting with 6-9, optional +91
    india_match = re.findall(r'(?:\+91[\s\-]?)?[6-9][0-9]{9}', text)
    if india_match:
        return india_match[0].strip()

    # Step 3: spaCy token — check for phone-like number tokens
    doc = nlp(text[:2000])
    for token in doc:
        if token.like_num:
            digits = re.sub(r'\D', '', token.text)
            if 10 <= len(digits) <= 15:
                return token.text.strip()

    # Step 4: General international number pattern
    general_match = re.findall(r'[\+\(]?[0-9][0-9\s\-\(\)]{8,}[0-9]', text)
    for m in general_match:
        digits = re.sub(r'\D', '', m)
        if 10 <= len(digits) <= 15:
            return m.strip()

    return "Not Found"

def extract_experience(text):
    # Finding years of experience mentioned in resume
    match = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)[\s\w]{0,15}experience', text, re.IGNORECASE)
    if match:
        return f"{match[0]} Year(s)"
    return "Not Mentioned"

def extract_education(text):
    # Searching for common education degree keywords
    degrees = ['B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'MBA', 'BCA', 'MCA',
               'Bachelor', 'Master', 'PhD', 'B.E', 'M.E', 'B.Com', 'M.Com']
    found = []
    for degree in degrees:
        if re.search(degree, text, re.IGNORECASE):
            found.append(degree)
    return ", ".join(found) if found else "Not Found"

def extract_skills(text):
    # Matching resume text against a predefined list of common tech skills
    skill_list = [
        'Python', 'Java', 'SQL', 'JavaScript', 'React', 'Node', 'HTML', 'CSS',
        'Machine Learning', 'Deep Learning', 'NLP', 'TensorFlow', 'Keras',
        'Pandas', 'NumPy', 'Scikit-learn', 'Git', 'Docker', 'AWS', 'Azure',
        'Workday', 'PeopleSoft', 'SAP', 'Oracle', 'MySQL', 'MongoDB',
        'Power BI', 'Tableau', 'Excel', 'C++', 'C#', 'Linux', 'Agile', 'Scrum',
        'REST API', 'Spring Boot', 'Django', 'Flask', 'Selenium', 'Postman'
    ]
    found = []
    for skill in skill_list:
        if re.search(skill, text, re.IGNORECASE):
            found.append(skill)
    return found

def get_word_stats(text):
    # Calculating total words, unique words, and sentences in resume text
    words      = text.split()
    sentences  = re.split(r'[.!?]', text)
    total_w    = len(words)
    unique_w   = len(set(words))
    total_s    = len([s for s in sentences if s.strip()])
    return total_w, unique_w, total_s

def get_top_keywords(cleaned_text, top_n=10):
    # Getting top N most frequent keywords from cleaned resume text
    words   = cleaned_text.split()
    counter = Counter(words)
    return counter.most_common(top_n)

# --- Full Analysis Function ---
def analyze_resume(raw_text, file_name):
    # Running all extraction and prediction steps together
    cleaned   = clean_resume_text(raw_text)
    category  = predict_category(cleaned)
    name      = extract_name(raw_text)
    email     = extract_email(raw_text)
    phone     = extract_phone(raw_text)
    experience= extract_experience(raw_text)
    education = extract_education(raw_text)
    skills    = extract_skills(raw_text)
    stats     = get_word_stats(raw_text)
    keywords  = get_top_keywords(cleaned)
    return {
        "file_name"  : file_name,
        "category"   : category,
        "name"       : name,
        "email"      : email,
        "phone"      : phone,
        "experience" : experience,
        "education"  : education,
        "skills"     : skills,
        "stats"      : stats,
        "keywords"   : keywords,
        "cleaned"    : cleaned,
        "timestamp"  : datetime.now().strftime("%H:%M:%S")
    }

# ============================================================
# App Interface (Python Streamlit Widgets Only)
# ============================================================

st.title("🚀  Resume Classifier")
st.divider()

# --- Sidebar ---
st.sidebar.title("📋 Project Info")
st.sidebar.info("""
**Team:** Group 4

**Model:** SVM + TF-IDF
""")

st.sidebar.title("📂 Supported Categories")
st.sidebar.write("💻 React Developer")
st.sidebar.write("🗄️ SQL Developer")
st.sidebar.write("☁️ Workday")
st.sidebar.write("🏢 Peoplesoft")

st.sidebar.divider()

# Sidebar: Upload History
st.sidebar.title("🕓 Upload History")
if st.session_state.history:
    for i, h in enumerate(reversed(st.session_state.history), 1):
        st.sidebar.write(f"**{i}.** {h['file_name']}")
        st.sidebar.caption(f"   → {h['category']} | {h['timestamp']}")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.caption("No uploads yet.")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["📁 Upload Resume", "✏️ Paste Text"])

# ================================================================
# Tab 1: File Upload
# ================================================================
with tab1:
    st.subheader("Upload a Resume File")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt']
    )

    if uploaded_file is not None:
        if model is not None:
            with st.spinner("Analyzing Resume..."):
                raw_text = get_text_from_file(uploaded_file)
                result   = analyze_resume(raw_text, uploaded_file.name)

                # Save to upload history in session state
                st.session_state.history.append({
                    "file_name": uploaded_file.name,
                    "category" : result["category"],
                    "timestamp": result["timestamp"]
                })

            # --- Prediction Result ---
            st.success(f"### ✅ Predicted Job Role: **{result['category']}**")
            st.metric(label="Predicted Category", value=result["category"])
            st.divider()

            # --- Candidate Details ---
            st.subheader("👤 Candidate Details")
            col1, col2 = st.columns(2)
            col1.write(f"**🧑 Name:**  {result['name']}")
            col1.write(f"**📧 Email:** {'⚠️ Not available in resume' if result['email'] == 'Not Found' else result['email']}")
            col1.write(f"**📞 Phone:** {'⚠️ Not available in resume' if result['phone'] == 'Not Found' else result['phone']}")
            col2.write(f"**🏫 Education:**  {'⚠️ Not mentioned' if result['education'] == 'Not Found' else result['education']}")
            col2.write(f"**💼 Experience:** {'⚠️ Not mentioned' if result['experience'] == 'Not Mentioned' else result['experience']}")

            col_a, col_b = st.columns(2)
            col_a.info(f"**File:** {uploaded_file.name}")
            col_b.info(f"**Size:** {round(uploaded_file.size / 1024, 2)} KB")
            st.divider()

            # --- Skills ---
            st.subheader("🛠️ Skills Extracted")
            if result["skills"]:
                skill_cols = st.columns(4)
                for idx, skill in enumerate(result["skills"]):
                    skill_cols[idx % 4].success(skill)
            else:
                st.warning("No matching skills found in resume.")
            st.divider()

            # --- Word Stats ---
            st.subheader("📊 Resume Stats")
            total_w, unique_w, total_s = result["stats"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Words",  total_w)
            c2.metric("Unique Words", unique_w)
            c3.metric("Sentences",    total_s)
            st.divider()

            # --- Bar Chart: Top Keywords ---
            st.subheader("📈 Top Keywords in Resume")
            if result["keywords"]:
                kw_words  = [k[0] for k in result["keywords"]]
                kw_counts = [k[1] for k in result["keywords"]]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(kw_words[::-1], kw_counts[::-1], color='#7c3aed')
                ax.set_xlabel("Frequency")
                ax.set_title("Top 10 Keywords")
                ax.set_facecolor("#f5f3ff")
                fig.patch.set_facecolor("#f5f3ff")
                plt.tight_layout()
                st.pyplot(fig)
            st.divider()

            # --- Processed Text Preview ---
            with st.expander("📄 View Processed Text"):
                st.write(result["cleaned"][:500] + "...")

        else:
            st.warning("Models are not loaded properly. Check your 'models' folder.")

# ================================================================
# Tab 2: Manual Text Input
# ================================================================
with tab2:
    st.subheader("Paste Resume Text Directly")
    manual_text = st.text_area(
        "Paste your resume text here:",
        height=250,
        placeholder="e.g. Experienced SQL Developer with 5 years in database design..."
    )

    if st.button("🔍 Classify Text"):
        if not manual_text.strip():
            st.warning("Pehle kuch text daalo.")
        elif model is None:
            st.warning("Models are not loaded properly. Check your 'models' folder.")
        else:
            with st.spinner("Analyzing Text..."):
                result = analyze_resume(manual_text, "Manual Input")
                st.session_state.history.append({
                    "file_name": "Manual Input",
                    "category" : result["category"],
                    "timestamp": result["timestamp"]
                })

            # --- Prediction Result ---
            st.success(f"### ✅ Predicted Job Role: **{result['category']}**")
            st.metric(label="Predicted Category", value=result["category"])
            st.divider()

            # --- Candidate Details ---
            st.subheader("👤 Candidate Details")
            col1, col2 = st.columns(2)
            col1.write(f"**🧑 Name:**  {result['name']}")
            col1.write(f"**📧 Email:** {'⚠️ Not available in resume' if result['email'] == 'Not Found' else result['email']}")
            col1.write(f"**📞 Phone:** {'⚠️ Not available in resume' if result['phone'] == 'Not Found' else result['phone']}")
            col2.write(f"**🏫 Education:**  {'⚠️ Not mentioned' if result['education'] == 'Not Found' else result['education']}")
            col2.write(f"**💼 Experience:** {'⚠️ Not mentioned' if result['experience'] == 'Not Mentioned' else result['experience']}")
            st.divider()

            # --- Skills ---
            st.subheader("🛠️ Skills Extracted")
            if result["skills"]:
                skill_cols = st.columns(4)
                for idx, skill in enumerate(result["skills"]):
                    skill_cols[idx % 4].success(skill)
            else:
                st.warning("No matching skills found.")
            st.divider()

            # --- Word Stats ---
            st.subheader("📊 Resume Stats")
            total_w, unique_w, total_s = result["stats"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Words",  total_w)
            c2.metric("Unique Words", unique_w)
            c3.metric("Sentences",    total_s)
            st.divider()

            # --- Bar Chart: Top Keywords ---
            st.subheader("📈 Top Keywords in Resume")
            if result["keywords"]:
                kw_words  = [k[0] for k in result["keywords"]]
                kw_counts = [k[1] for k in result["keywords"]]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(kw_words[::-1], kw_counts[::-1], color='#7c3aed')
                ax.set_xlabel("Frequency")
                ax.set_title("Top 10 Keywords")
                ax.set_facecolor("#f5f3ff")
                fig.patch.set_facecolor("#f5f3ff")
                plt.tight_layout()
                st.pyplot(fig)
            st.divider()

            # --- Processed Text Preview ---
            with st.expander("📄 View Processed Text"):
                st.write(result["cleaned"][:500] + "...")

# --- Footer ---
st.divider()
st.write(f"© {datetime.now().year} | Developed by Group 4")