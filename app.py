import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import lime
import lime.lime_text
import shap
import io
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import pytesseract

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="AI-Powered Grader")

# --- 1. FILE PARSING LOGIC ---
# Function to extract text from various file formats
def extract_text(uploaded_file):
    """Extracts text from PDF, DOCX, TXT, and image files."""
    if uploaded_file.name.endswith('.pdf'):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode("utf-8")
    return "" # Return empty string for unsupported formats

# --- 2. MULTI-MODEL GRADING LOGIC ---
# We simulate different models by using different keyword-based rubrics.
# In a real-world scenario, each of these would be a separately trained model.

# --- Model 1: Essay Grading ---
@st.cache_resource
def get_essay_model():
    keywords = {
        'A': ['innovative', 'comprehensive', 'insightful', 'thorough', 'exemplary', 'deep analysis'],
        'B': ['well-written', 'organized', 'relevant', 'sufficient detail'],
        'C': ['adequate', 'basic', 'superficial', 'lacks depth'],
        'F': ['incomplete', 'off-topic', 'major errors', 'poorly written']
    }
    data = [{'text': ' '.join(np.random.choice(v, 5)) + ' filler text ' * 10, 'grade': k} for k, v in keywords.items() for _ in range(50)]
    df = pd.DataFrame(data)
    pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
    pipeline.fit(df['text'], df['grade'])
    return pipeline, df['grade'].unique()

# --- Model 2: Resume Grading ---
@st.cache_resource
def get_resume_model():
    keywords = {
        'Strong': ['managed', 'led', 'developed', 'achieved', 'quantifiable', 'experience', 'skills', 'certified'],
        'Average': ['responsible for', 'assisted', 'duties included', 'team player', 'hardworking'],
        'Weak': ['entry-level', 'learning', 'aspiring', 'no experience', 'references available upon request']
    }
    data = [{'text': ' '.join(np.random.choice(v, 5)) + ' filler resume text ' * 10, 'grade': k} for k, v in keywords.items() for _ in range(50)]
    df = pd.DataFrame(data)
    pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
    pipeline.fit(df['text'], df['grade'])
    return pipeline, df['grade'].unique()

# --- Model 3: Python Code Grading (Static Analysis Simulation) ---
@st.cache_resource
def get_code_model():
    keywords = {
        'Excellent': ['class', 'def', 'return', 'import', '# comments', 'try:', 'except:'],
        'Good': ['for loop', 'while loop', 'if/else', 'print'],
        'Needs Improvement': ['magic numbers', 'hardcoded', 'global variable', 'no comments']
    }
    # This is a vast oversimplification. Real code grading uses linters, ASTs, and unit tests.
    data = [{'text': ' '.join(np.random.choice(v, 4)) + ' # python code ' * 10, 'grade': k} for k, v in keywords.items() for _ in range(50)]
    df = pd.DataFrame(data)
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(df['text'], df['grade'])
    return pipeline, df['grade'].unique()

# Load all models
essay_pipeline, essay_classes = get_essay_model()
resume_pipeline, resume_classes = get_resume_model()
code_pipeline, code_classes = get_code_model()

# --- 3. XAI EXPLAINER SETUP ---
# We will create explainers dynamically based on the selected model.

# --- 4. STREAMLIT UI ---
st.title("Universal AI Grader & Explainer 📝📄💻")
st.markdown("This tool grades different types of documents and explains its reasoning using LIME and SHAP.")

# Content type selector
content_type = st.selectbox(
    "1. Choose the type of content you want to grade:",
    ("Essay / Assignment", "Resume", "Python Code", "Research Paper (beta)")
)

# Set the active model based on user selection
if content_type == "Essay / Assignment" or content_type == "Research Paper (beta)":
    model_pipeline = essay_pipeline
    class_names = essay_classes
elif content_type == "Resume":
    model_pipeline = resume_pipeline
    class_names = resume_classes
elif content_type == "Python Code":
    model_pipeline = code_pipeline
    class_names = code_classes

st.markdown("---")

# Input method tabs
st.subheader("2. Provide the content")
tab1, tab2 = st.tabs(["📋 Paste Text", "📤 Upload File"])

user_content = ""
with tab1:
    user_content = st.text_area("Paste your content here:", height=300)

with tab2:
    uploaded_file = st.file_uploader(
        "Upload a file (.pdf, .docx, .txt, .jpg, .png)",
        type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png']
    )
    if uploaded_file:
        with st.spinner("Extracting text from file..."):
            user_content = extract_text(uploaded_file)
            st.text_area("Extracted Text:", user_content, height=200, disabled=True)

st.markdown("---")

if st.button("Grade and Explain", type="primary"):
    if not user_content.strip():
        st.warning("Please paste text or upload a file to proceed.")
    else:
        # --- 5. PREDICTION AND EXPLANATION ---
        with st.spinner("AI is grading the document..."):
            prediction = model_pipeline.predict([user_content])[0]
            prediction_proba = model_pipeline.predict_proba([user_content])

            st.header(f"AI-Generated Grade: **{prediction}**")
            proba_df = pd.DataFrame(prediction_proba, columns=class_names, index=['Probability'])
            st.dataframe(proba_df)
            
            # --- LIME Explanation ---
            st.subheader("LIME Explanation")
            explainer_lime = lime.lime_text.LimeTextExplainer(class_names=class_names)
            exp_lime = explainer_lime.explain_instance(
                user_content, model_pipeline.predict_proba, num_features=10,
                labels=[list(class_names).index(prediction)]
            )
            st.components.v1.html(exp_lime.as_html(), height=400)
            
            # --- SHAP Explanation ---
            st.subheader("SHAP Explanation")
            # SHAP explainer needs to be created with the correct model pipeline
            def f_shap(texts):
                return model_pipeline.predict_proba(texts)
            
            masker = shap.maskers.Text(r'\W+')
            explainer_shap = shap.Explainer(f_shap, masker, output_names=list(class_names))
            shap_values = explainer_shap([user_content])
            predicted_class_index = list(class_names).index(prediction)
            
            plot_object = shap.force_plot(shap_values[0, :, predicted_class_index])
            shap_html_file = io.StringIO()
            shap.save_html(shap_html_file, plot_object)
            st.components.v1.html(shap_html_file.getvalue(), height=200)