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

def extract_skills(text):
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
# NEW: Helper — Convert results list to CSV bytes
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
# Tab 2: NEW — Bulk Resume Upload
# ================================================================
with tab2:
    st.subheader("📦 Bulk Resume Upload")
    st.info("Upload multiple resumes at once — all results will appear in a single table.")

    bulk_files = st.file_uploader(
        "Upload multiple resumes (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key="bulk_uploader"
    )

    if bulk_files:
        if model is None:
            st.warning("Models are not loaded properly. Check your 'models' folder.")
        else:
            if st.button("🚀 Analyze All Resumes"):
                new_results = []
                progress_bar = st.progress(0)
                status_text  = st.empty()

                for i, file in enumerate(bulk_files):
                    status_text.text(f"Analyzing: {file.name} ({i+1}/{len(bulk_files)})")
                    try:
                        raw_text = get_text_from_file(file)
                        result   = analyze_resume(raw_text, file.name)
                        new_results.append(result)
                        st.session_state.bulk_results.append(result)
                        st.session_state.history.append({
                            "file_name": file.name,
                            "category" : result["category"],
                            "timestamp": result["timestamp"]
                        })
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                    progress_bar.progress((i + 1) / len(bulk_files))

                status_text.success(f"✅ {len(new_results)} resumes analyzed successfully!")

                # Show quick results table
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


# ================================================================
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
