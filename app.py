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
import io
import zipfile
import tempfile

# --- Page Settings (MUST be first Streamlit command) ---
st.set_page_config(page_title="Group 4 | Resume AI", page_icon="🧠", layout="wide")

# --- Stopwords Setup ---
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- spaCy NLP Model Load ---
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# Session state for storing bulk results table data
if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = []

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/svm_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        le    = joblib.load('models/label_encoder.pkl')
        return model, tfidf, le
    except Exception as e:
        st.error(f"Model files not found: {e}")
        return None, None, None

model, tfidf, le = load_models()

# --- File Extraction ---
def get_text_from_file(file):
    if file.name.endswith('.docx'):
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    elif file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return file.read().decode('utf-8', errors='ignore')

# --- Text Cleaning ---
def clean_resume_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Prediction ---
def predict_category(cleaned_text):
    vectorized = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized)
    category = le.inverse_transform(prediction)[0]
    if category == "General_Developer":
        category = "React Developer"
    return category

# --- Extraction Functions ---
def extract_name(text):
    label_match = re.search(r'(?:Name\s*[:\-]\s*)([A-Za-z]+(?: [A-Za-z]+){1,3})', text, re.IGNORECASE)
    if label_match:
        return label_match.group(1).strip().title()
    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if 2 <= len(name.split()) <= 4:
                return name.title()
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
    label_match = re.search(
        r'(?:e[\-]?mail|mail|contact)\s*[:\-]\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
        text, re.IGNORECASE
    )
    if label_match:
        return label_match.group(1).strip()
    doc = nlp(text[:2000])
    for token in doc:
        if token.like_email:
            return token.text.strip()
    match = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    valid = [m for m in match if not re.search(r'\.(png|jpg|jpeg|gif|svg|pdf|docx)$', m, re.IGNORECASE)]
    return valid[0] if valid else "Not Found"

def extract_phone(text):
    label_match = re.search(
        r'(?:phone|mobile|contact|cell|ph|tel|mob)\s*[:\-]?\s*([\+\(]?[0-9][0-9\s\-\(\)]{8,}[0-9])',
        text, re.IGNORECASE
    )
    if label_match:
        digits = re.sub(r'\D', '', label_match.group(1))
        if 10 <= len(digits) <= 15:
            return label_match.group(1).strip()
    india_match = re.findall(r'(?:\+91[\s\-]?)?[6-9][0-9]{9}', text)
    if india_match:
        return india_match[0].strip()
    doc = nlp(text[:2000])
    for token in doc:
        if token.like_num:
            digits = re.sub(r'\D', '', token.text)
            if 10 <= len(digits) <= 15:
                return token.text.strip()
    general_match = re.findall(r'[\+\(]?[0-9][0-9\s\-\(\)]{8,}[0-9]', text)
    for m in general_match:
        digits = re.sub(r'\D', '', m)
        if 10 <= len(digits) <= 15:
            return m.strip()
    return "Not Found"

def extract_experience(text):
    match = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)[\s\w]{0,15}experience', text, re.IGNORECASE)
    if match:
        return f"{match[0]} Year(s)"
    return "Not Mentioned"

def extract_education(text):
    degrees = ['B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'MBA', 'BCA', 'MCA',
               'Bachelor', 'Master', 'PhD', 'B.E', 'M.E', 'B.Com', 'M.Com']
    found = []
    for degree in degrees:
        if re.search(degree, text, re.IGNORECASE):
            found.append(degree)
    return ", ".join(found) if found else "Not Found"

# All skills organized by category — used for display in candidate profile
ALL_SKILLS = {
    # --- SQL Developer Skills ---
    "SQL Developer": [
        'SQL', 'MySQL', 'Oracle', 'MongoDB', 'PostgreSQL', 'SSMS',
        'Stored Procedure', 'NoSQL', 'ETL', 'Data Warehouse',
        'Power BI', 'Tableau', 'Excel', 'Python', 'Azure',
        'AWS', 'Linux', 'Git', 'Agile', 'Query'
    ],
    # --- React Developer Skills ---
    "React Developer": [
        'React', 'JavaScript', 'HTML', 'CSS', 'Node', 'Redux',
        'TypeScript', 'REST API', 'JSON', 'npm', 'Git', 'Docker',
        'AWS', 'Azure', 'Python', 'Agile', 'Scrum',
        'Jest', 'Webpack', 'Bootstrap'
    ],
    # --- Workday Skills ---
    "Workday": [
        'Workday', 'HCM', 'HRIS', 'Payroll', 'ERP',
        'Business Process', 'Workday Studio', 'Integration', 'BIRT', 'Absence',
        'SAP', 'Oracle', 'PeopleSoft', 'Excel', 'SQL',
        'Python', 'Agile', 'Reporting', 'Compensation', 'Recruiting'
    ],
    # --- Peoplesoft Skills ---
    "Peoplesoft": [
        'PeopleSoft', 'PeopleCode', 'Application Engine', 'SQR', 'Component Interface',
        'HCM', 'FSCM', 'ERP', 'Integration Broker', 'PeopleSoft Query',
        'Oracle', 'SQL', 'Workday', 'SAP', 'Excel',
        'Python', 'Agile', 'COBOL', 'Unix', 'Reporting'
    ],
}

# Flat list of all unique skills for general extraction
_ALL_SKILLS_FLAT = list({skill for skills in ALL_SKILLS.values() for skill in skills})

def extract_skills(text, category=None):
    # If category is known, use category-specific skill list for more accurate extraction
    if category and category in ALL_SKILLS:
        skill_list = ALL_SKILLS[category]
    else:
        skill_list = _ALL_SKILLS_FLAT
    found = []
    for skill in skill_list:
        if re.search(re.escape(skill), text, re.IGNORECASE):
            found.append(skill)
    return found

def get_word_stats(text):
    words      = text.split()
    sentences  = re.split(r'[.!?]', text)
    total_w    = len(words)
    unique_w   = len(set(words))
    total_s    = len([s for s in sentences if s.strip()])
    return total_w, unique_w, total_s

def get_top_keywords(cleaned_text, top_n=10):
    words   = cleaned_text.split()
    counter = Counter(words)
    return counter.most_common(top_n)

# --- Full Analysis Function ---
def analyze_resume(raw_text, file_name):
    cleaned   = clean_resume_text(raw_text)
    category  = predict_category(cleaned)
    name      = extract_name(raw_text)
    email     = extract_email(raw_text)
    phone     = extract_phone(raw_text)
    experience= extract_experience(raw_text)
    education = extract_education(raw_text)
    skills    = extract_skills(raw_text, category)   # category-specific skill extraction
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
# Category-Specific Skill Criteria
# Each job category has its own required and bonus skills
# ============================================================
# ============================================================
# Category-Specific Skill Criteria (Expanded)
# Each category has required + bonus skills list
# ============================================================
CATEGORY_SKILLS = {
    "SQL Developer": {
        "required": ["SQL", "MySQL", "Oracle", "MongoDB", "PostgreSQL",
                     "Database", "Query", "Stored Procedure", "NoSQL", "SSMS"],
        "bonus"   : ["Python", "Power BI", "Tableau", "Excel", "Azure",
                     "AWS", "Linux", "Git", "ETL", "Data Warehouse", "Agile"],
    },
    "React Developer": {
        "required": ["React", "JavaScript", "HTML", "CSS", "Node",
                     "Redux", "TypeScript", "REST API", "JSON", "npm"],
        "bonus"   : ["Git", "Docker", "AWS", "Azure", "Python",
                     "Agile", "Scrum", "Jest", "Webpack", "Bootstrap"],
    },
    "Workday": {
        "required": ["Workday", "HCM", "HRIS", "Payroll", "ERP",
                     "Business Process", "Workday Studio", "Integration", "BIRT", "Absence"],
        "bonus"   : ["SAP", "Oracle", "PeopleSoft", "Excel", "SQL",
                     "Python", "Agile", "Reporting", "Compensation", "Recruiting"],
    },
    "Peoplesoft": {
        "required": ["PeopleSoft", "PeopleCode", "Application Engine", "SQR", "Component Interface",
                     "HCM", "FSCM", "ERP", "Integration Broker", "PeopleSoft Query"],
        "bonus"   : ["Oracle", "SQL", "Workday", "SAP", "Excel",
                     "Python", "Agile", "COBOL", "Unix", "Reporting"],
    },
}

# ============================================================
# Resume Scoring & Ranking Function (Category-Based)
# Score is calculated out of 100:
#   - Required Skills : up to 50 pts — percentage of required skills matched * 50
#   - Bonus Skills    : up to 20 pts — percentage of bonus skills matched * 20
#   - Experience      : up to 15 pts (3 pts per year, max 5 yrs)
#   - Education       : up to 10 pts (based on degree level)
#   - Contact Info    : up to  5 pts (email 3pts + phone 2pts)
# ============================================================
def calculate_score(result):
    category = result["category"]
    # Use both raw text fields for better skill matching
    raw_text = (result.get("cleaned", "") + " " +
                result.get("education", "") + " " +
                " ".join(result.get("skills", [])))
    criteria = CATEGORY_SKILLS.get(category, {
        "required": ["Python", "SQL", "Git"],
        "bonus"   : ["Excel", "Agile", "Docker"]
    })

    # --- Required Skills Score (max 50) ---
    # Score = (matched / total_required) * 50
    matched_required = [s for s in criteria["required"] if re.search(s, raw_text, re.IGNORECASE)]
    total_req = len(criteria["required"])
    req_score = round((len(matched_required) / total_req) * 50) if total_req > 0 else 0

    # --- Bonus Skills Score (max 20) ---
    # Score = (matched / total_bonus) * 20
    matched_bonus = [s for s in criteria["bonus"] if re.search(s, raw_text, re.IGNORECASE)]
    total_bonus = len(criteria["bonus"])
    bonus_score = round((len(matched_bonus) / total_bonus) * 20) if total_bonus > 0 else 0

    skill_score = req_score + bonus_score

    # --- Experience Score (max 15) ---
    exp = result["experience"]
    exp_score = 0
    if exp != "Not Mentioned":
        m = re.search(r'(\d+)', exp)
        if m:
            years = int(m.group(1))
            exp_score = min(years * 3, 15)

    # --- Education Score (max 10) ---
    edu = result["education"].lower()
    edu_score = 0
    if any(d in edu for d in ['phd', 'doctorate']):
        edu_score = 10
    elif any(d in edu for d in ['m.tech', 'm.sc', 'mca', 'mba', 'm.e', 'm.com', 'master']):
        edu_score = 8
    elif any(d in edu for d in ['b.tech', 'b.sc', 'bca', 'b.e', 'b.com', 'bachelor']):
        edu_score = 6
    elif edu != "not found":
        edu_score = 4

    # --- Contact Info Score (max 5) ---
    contact_score = 0
    if result["email"] != "Not Found":
        contact_score += 3
    if result["phone"] != "Not Found":
        contact_score += 2

    total = min(req_score + bonus_score + exp_score + edu_score + contact_score, 100)
    return total, skill_score, matched_required, matched_bonus, exp_score, edu_score, contact_score

def get_grade(score):
    # Assign letter grade based on total score out of 100
    if score >= 75:
        return "🥇 A+"
    elif score >= 60:
        return "🥈 A"
    elif score >= 45:
        return "🥉 B+"
    elif score >= 30:
        return "📄 B"
    else:
        return "📋 C"

# ============================================================
# Helper — Convert results list to CSV bytes
# ============================================================
def results_to_csv(results_list):
    rows = []
    for r in results_list:
        total_w, unique_w, total_s = r["stats"]
        rows.append({
            "File Name"  : r["file_name"],
            "Name"       : r["name"],
            "Email"      : r["email"],
            "Phone"      : r["phone"],
            "Category"   : r["category"],
            "Experience" : r["experience"],
            "Education"  : r["education"],
            "Skills"     : ", ".join(r["skills"]),
            "Total Words": total_w,
            "Sentences"  : total_s,
            "Timestamp"  : r["timestamp"],
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')

# ============================================================
# App Interface
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
        st.session_state.bulk_results = []
        st.rerun()
else:
    st.sidebar.caption("No uploads yet.")

# --- Tab Layout --- (NEW: 4 tabs)
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Upload Resume",
    "📦 Bulk Upload",
    "📊 Summary Table",
    "✏️ Paste Text"
])

# ================================================================
# Tab 1: Single File Upload
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

                st.session_state.history.append({
                    "file_name": uploaded_file.name,
                    "category" : result["category"],
                    "timestamp": result["timestamp"]
                })
                # Also add to bulk_results for summary table
                st.session_state.bulk_results.append(result)

            st.success(f"### ✅ Predicted Job Role: **{result['category']}**")
            st.metric(label="Predicted Category", value=result["category"])
            st.divider()

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

            st.subheader("🛠️ Skills Extracted")
            if result["skills"]:
                skill_cols = st.columns(4)
                for idx, skill in enumerate(result["skills"]):
                    skill_cols[idx % 4].success(skill)
            else:
                st.warning("No matching skills found in resume.")
            st.divider()

            st.subheader("📊 Resume Stats")
            total_w, unique_w, total_s = result["stats"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Words",  total_w)
            c2.metric("Unique Words", unique_w)
            c3.metric("Sentences",    total_s)
            st.divider()

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

            with st.expander("📄 View Processed Text"):
                st.write(result["cleaned"][:500] + "...")

            # --- Download Single Result ---
            csv_data = results_to_csv([result])
            st.download_button(
                label="⬇️ Download This Result as CSV",
                data=csv_data,
                file_name=f"result_{uploaded_file.name}_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Models are not loaded properly. Check your 'models' folder.")


# ================================================================
# Helper: extract files from ZIP
# ================================================================
def extract_files_from_zip(zip_file):
    """
    Extract PDF, DOCX, TXT files from a ZIP archive.
    Handles nested folders at any depth.
    Returns (extracted_list, skipped_list).
    Each extracted item is a BytesIO with .name (filename) and .path (full ZIP path).
    """
    extracted = []
    skipped   = []

    with zipfile.ZipFile(zip_file, 'r') as z:
        for name in z.namelist():
            base = name.split('/')[-1]
            # Skip folders and system/hidden files
            if name.endswith('/') or base.startswith('__') or base.startswith('.') or base == '':
                continue
            ext = base.lower().rsplit('.', 1)[-1] if '.' in base else ''
            if ext in ['pdf', 'docx', 'txt']:
                data     = z.read(name)
                buf      = io.BytesIO(data)
                buf.name = base    # filename only — used by get_text_from_file
                buf.path = name    # full path inside ZIP — shown in UI
                extracted.append(buf)
            else:
                if ext:
                    skipped.append(name)

    return extracted, skipped

# ================================================================
# Tab 2: Bulk Resume Upload (Files + ZIP + Category Filter)
# ================================================================
with tab2:
    st.subheader("📦 Bulk Resume Upload")
    st.info("Upload multiple resumes (PDF/DOCX/TXT) or a ZIP file/folder archive — all results appear in a ranked table.")

    upload_mode = st.radio(
        "Select upload type:",
        ["📄 Individual Files", "🗜️ ZIP File / Folder Archive"],
        horizontal=True
    )

    bulk_files_raw = []

    if upload_mode == "📄 Individual Files":
        uploaded_bulk = st.file_uploader(
            "Upload multiple resumes (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="bulk_uploader"
        )
        if uploaded_bulk:
            bulk_files_raw = uploaded_bulk

    else:
        zip_file = st.file_uploader(
            "Upload a ZIP file containing resumes",
            type=['zip'],
            key="zip_uploader"
        )
        if zip_file:
            try:
                extracted, skipped = extract_files_from_zip(zip_file)
                if extracted:
                    st.success(f"✅ Found **{len(extracted)}** resume(s) in ZIP (including nested folders)")

                    # Show folder tree structure
                    with st.expander("📂 View ZIP Contents (folder structure)", expanded=True):
                        # Group files by their folder path
                        folder_map = {}
                        for f in extracted:
                            parts = f.path.split('/')
                            folder = '/'.join(parts[:-1]) if len(parts) > 1 else '(root)'
                            folder_map.setdefault(folder, []).append(f.name)

                        for folder, files in folder_map.items():
                            st.markdown(f"**📁 {folder}**")
                            for fname in files:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 {fname}")

                        if skipped:
                            st.caption(f"⚠️ Skipped {len(skipped)} unsupported file(s): " +
                                       ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""))

                    bulk_files_raw = extracted
                else:
                    st.warning("No PDF, DOCX, or TXT files found inside the ZIP.")
                    if skipped:
                        st.caption(f"Found these unsupported files: {', '.join(skipped[:10])}")
            except Exception as e:
                st.error(f"Could not read ZIP file: {e}")

    # --- Category Filter for Ranking ---
    if bulk_files_raw:
        st.divider()
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            selected_categories = st.multiselect(
                "🔍 Filter ranking by category (leave empty = show all):",
                options=["SQL Developer", "React Developer", "Workday", "Peoplesoft"],
                default=[],
                help="Select one or more categories to compare only those resumes against each other"
            )
        with filter_col2:
            st.write("")
            st.write("")
            analyze_btn = st.button("🚀 Analyze All Resumes", use_container_width=True)

        if model is None:
            st.warning("Models are not loaded properly. Check your 'models' folder.")
        elif analyze_btn:
            new_results = []
            progress_bar = st.progress(0)
            status_text  = st.empty()

            for i, file in enumerate(bulk_files_raw):
                # Use full ZIP path (folder/file.pdf) if available, else just filename
                display_name = getattr(file, 'path', file.name)
                status_text.text(f"Analyzing: {file.name} ({i+1}/{len(bulk_files_raw)})")
                try:
                    raw_text = get_text_from_file(file)
                    result   = analyze_resume(raw_text, display_name)
                    new_results.append(result)
                    st.session_state.bulk_results.append(result)
                    st.session_state.history.append({
                        "file_name": display_name,
                        "category" : result["category"],
                        "timestamp": result["timestamp"]
                    })
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                progress_bar.progress((i + 1) / len(bulk_files_raw))

            status_text.success(f"✅ {len(new_results)} resumes analyzed successfully!")


            # Show quick results table + ranking
            if new_results:
                st.subheader("📋 Bulk Analysis Results")
                rows = []
                for r in new_results:
                    rows.append({
                        "File"      : r["file_name"],
                        "Name"      : r["name"],
                        "Category"  : r["category"],
                        "Email"     : r["email"],
                        "Phone"     : r["phone"],
                        "Experience": r["experience"],
                        "Education" : r["education"],
                        "Skills"    : ", ".join(r["skills"][:5]) + ("..." if len(r["skills"]) > 5 else ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.divider()

                # --- Apply category filter for ranking ---
                if selected_categories:
                    rank_results = [r for r in new_results if r["category"] in selected_categories]
                    st.info(f"🔍 Ranking filtered for: **{', '.join(selected_categories)}** — {len(rank_results)} candidate(s)")
                else:
                    rank_results = new_results
                    st.info(f"📋 Showing ranking for all {len(rank_results)} candidates across all categories.")

                if not rank_results:
                    st.warning("No resumes matched the selected categories.")
                else:
                    st.subheader("🏆 Candidate Ranking")
                    st.caption("Ranked by: Required Skills (50pts) + Bonus Skills (20pts) + Experience (15pts) + Education (10pts) + Contact (5pts) = 100pts")

                    ranking_rows = []
                    for r in rank_results:
                        total_score, skill_score, matched_req, matched_bonus, exp_score, edu_score, contact_score = calculate_score(r)
                        ranking_rows.append({
                            "_score"           : total_score,
                            "Name"             : r["name"],
                            "File"             : r["file_name"],
                            "Category"         : r["category"],
                            "Grade"            : get_grade(total_score),
                            "Required Skills"  : ", ".join(matched_req) if matched_req else "None",
                            "Bonus Skills"     : ", ".join(matched_bonus) if matched_bonus else "None",
                            "Skills pts"       : f"{skill_score} / 70",
                            "Experience pts"   : f"{exp_score} / 15",
                            "Education pts"    : f"{edu_score} / 10",
                            "Contact pts"      : f"{contact_score} / 5",
                        })

                    ranking_rows.sort(key=lambda x: x["_score"], reverse=True)

                    for i, row in enumerate(ranking_rows, 1):
                        row["Rank"]        = i
                        row["Score / 100"] = row["_score"]
                        del row["_score"]

                    df_rank = pd.DataFrame(ranking_rows)[[
                        "Rank", "Name", "File", "Category", "Grade",
                        "Score / 100", "Required Skills", "Bonus Skills",
                        "Skills pts", "Experience pts", "Education pts", "Contact pts"
                    ]]
                    st.dataframe(df_rank, use_container_width=True)

                    # Top 3 candidates
                    if len(ranking_rows) >= 1:
                        st.divider()
                        st.subheader("🎖️ Top Candidates")
                        medals = ["🥇", "🥈", "🥉"]
                        top_n = min(3, len(ranking_rows))
                        cols = st.columns(top_n)
                        for i in range(top_n):
                            r = ranking_rows[i]
                            cols[i].metric(
                                label=f"{medals[i]} Rank {i+1}",
                                value=r["Name"],
                                delta=f"{r['Score / 100']} / 100  |  {r['Grade']}"
                            )

                    # Score bar chart
                    st.divider()
                    st.subheader("📊 Score Comparison")
                    names  = [r["Name"] if r["Name"] != "Not Found" else r["File"] for r in ranking_rows]
                    scores = [r["Score / 100"] for r in ranking_rows]
                    colors = ['#7c3aed' if i == 0 else '#a78bfa' if i == 1 else '#c4b5fd' if i == 2 else '#ede9fe' for i in range(len(scores))]
                    fig_rank, ax_rank = plt.subplots(figsize=(10, max(4, len(names) * 0.7)))
                    bars = ax_rank.barh(names[::-1], scores[::-1], color=colors[::-1])
                    ax_rank.set_xlabel("Score (out of 100)")
                    ax_rank.set_title("Candidate Score Ranking")
                    ax_rank.set_xlim(0, 100)
                    ax_rank.set_facecolor("#f5f3ff")
                    fig_rank.patch.set_facecolor("#f5f3ff")
                    for bar, score in zip(bars, scores[::-1]):
                        ax_rank.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                                     f'{score}', va='center', fontweight='bold', color='#7c3aed')
                    plt.tight_layout()
                    st.pyplot(fig_rank)
                    st.divider()

                    # Category distribution chart
                    st.subheader("📊 Category Distribution")
                    categories = [r["category"] for r in new_results]
                    cat_counts = Counter(categories)
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.bar(cat_counts.keys(), cat_counts.values(), color='#7c3aed')
                    ax2.set_ylabel("Count")
                    ax2.set_title("Resumes by Category")
                    ax2.set_facecolor("#f5f3ff")
                    fig2.patch.set_facecolor("#f5f3ff")
                    plt.tight_layout()
                    st.pyplot(fig2)

                    # Download bulk results
                    csv_bulk = results_to_csv(new_results)
                    st.download_button(
                        label="⬇️ Download Bulk Results as CSV",
                        data=csv_bulk,
                        file_name=f"bulk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )


# Tab 3: NEW — Summary Table (all analyzed so far)
# ================================================================
with tab3:
    st.subheader("📊 Summary Table — All Analyzed Resumes")

    if not st.session_state.bulk_results:
        st.info("No resumes analyzed yet. Please upload resumes from Tab 1 or Tab 2.")
    else:
        results_list = st.session_state.bulk_results

        # --- KPI Cards ---
        total_analyzed = len(results_list)
        categories     = [r["category"] for r in results_list]
        cat_counts     = Counter(categories)
        most_common    = cat_counts.most_common(1)[0][0]

        k1, k2, k3 = st.columns(3)
        k1.metric("📄 Total Resumes Analyzed", total_analyzed)
        k2.metric("🏆 Most Common Category",    most_common)
        k3.metric("📂 Unique Categories",        len(cat_counts))
        st.divider()

        # --- Full Summary Table ---
        st.subheader("📋 Detailed Summary")
        rows = []
        for r in results_list:
            total_w, unique_w, total_s = r["stats"]
            rows.append({
                "File"        : r["file_name"],
                "Name"        : r["name"],
                "Category"    : r["category"],
                "Email"       : r["email"],
                "Phone"       : r["phone"],
                "Experience"  : r["experience"],
                "Education"   : r["education"],
                "Skills"      : ", ".join(r["skills"][:5]) + ("..." if len(r["skills"]) > 5 else ""),
                "Total Words" : total_w,
                "Sentences"   : total_s,
                "Time"        : r["timestamp"],
            })
        df_summary = pd.DataFrame(rows)

        # Filter by category
        all_cats = ["All"] + sorted(df_summary["Category"].unique().tolist())
        filter_cat = st.selectbox("🔍 Filter by Category", all_cats)
        if filter_cat != "All":
            df_filtered = df_summary[df_summary["Category"] == filter_cat]
        else:
            df_filtered = df_summary

        st.dataframe(df_filtered, use_container_width=True)
        st.caption(f"Showing {len(df_filtered)} of {len(df_summary)} resumes")
        st.divider()

        # --- Category Pie Chart ---
        st.subheader("🥧 Category Breakdown")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.pie(
            cat_counts.values(),
            labels=cat_counts.keys(),
            autopct='%1.1f%%',
            colors=['#7c3aed', '#a78bfa', '#c4b5fd', '#ede9fe'],
            startangle=140
        )
        ax3.set_facecolor("#f5f3ff")
        fig3.patch.set_facecolor("#f5f3ff")
        st.pyplot(fig3)
        st.divider()

        # --- Download All Results ---
        st.subheader("⬇️ Download Results")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            csv_all = results_to_csv(results_list)
            st.download_button(
                label="📥 Download All Results (CSV)",
                data=csv_all,
                file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_d2:
            if filter_cat != "All":
                filtered_results = [r for r in results_list if r["category"] == filter_cat]
                csv_filtered = results_to_csv(filtered_results)
                st.download_button(
                    label=f"📥 Download '{filter_cat}' Only (CSV)",
                    data=csv_filtered,
                    file_name=f"{filter_cat.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Select a category filter above to download filtered CSV.")

        st.divider()
        if st.button("🗑️ Clear All Results"):
            st.session_state.bulk_results = []
            st.session_state.history = []
            st.rerun()


# ================================================================
# Tab 4: Manual Text Input (was Tab 2)
# ================================================================
with tab4:
    st.subheader("Paste Resume Text Directly")
    manual_text = st.text_area(
        "Paste your resume text here:",
        height=250,
        placeholder="e.g. Experienced SQL Developer with 5 years in database design..."
    )

    if st.button("🔍 Classify Text"):
        if not manual_text.strip():
            st.warning("Please enter some text first.")
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
                st.session_state.bulk_results.append(result)

            st.success(f"### ✅ Predicted Job Role: **{result['category']}**")
            st.metric(label="Predicted Category", value=result["category"])
            st.divider()

            st.subheader("👤 Candidate Details")
            col1, col2 = st.columns(2)
            col1.write(f"**🧑 Name:**  {result['name']}")
            col1.write(f"**📧 Email:** {'⚠️ Not available in resume' if result['email'] == 'Not Found' else result['email']}")
            col1.write(f"**📞 Phone:** {'⚠️ Not available in resume' if result['phone'] == 'Not Found' else result['phone']}")
            col2.write(f"**🏫 Education:**  {'⚠️ Not mentioned' if result['education'] == 'Not Found' else result['education']}")
            col2.write(f"**💼 Experience:** {'⚠️ Not mentioned' if result['experience'] == 'Not Mentioned' else result['experience']}")
            st.divider()

            st.subheader("🛠️ Skills Extracted")
            if result["skills"]:
                skill_cols = st.columns(4)
                for idx, skill in enumerate(result["skills"]):
                    skill_cols[idx % 4].success(skill)
            else:
                st.warning("No matching skills found.")
            st.divider()

            st.subheader("📊 Resume Stats")
            total_w, unique_w, total_s = result["stats"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Words",  total_w)
            c2.metric("Unique Words", unique_w)
            c3.metric("Sentences",    total_s)
            st.divider()

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

            with st.expander("📄 View Processed Text"):
                st.write(result["cleaned"][:500] + "...")

            # Download single result
            csv_data = results_to_csv([result])
            st.download_button(
                label="⬇️ Download This Result as CSV",
                data=csv_data,
                file_name=f"manual_result_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv"
            )

# --- Footer ---
st.divider()
st.write(f"© {datetime.now().year} | Developed by Group 4")