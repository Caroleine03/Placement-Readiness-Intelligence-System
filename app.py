import streamlit as st
import pypdf
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from groq import Groq
import joblib
import json
import pandas as pd  
from fpdf import FPDF 

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

model = joblib.load('logistic_model.joblib')  

feature_order = ['skill_match_percentage', 'critical_skill_match_percentage',
     'project_relevance_score', 'certification_relevance_score', 
     'internship_relevance_score', 'resume_completeness_score', 
     'keyword_match_score', 'role_category_match_score', 
     'missing_skills_count','critical_missing_skills_count']

bucket_a_cols = ['skill_match_percentage', 'critical_skill_match_percentage','project_relevance_score', 
                 'certification_relevance_score', 'internship_relevance_score', 'resume_completeness_score', 
                 'keyword_match_score', 'role_category_match_score']

def analyze_resume(resume_text, jd_text):
    # Step 1: build the prompt (reuse your JSON template, just swap {text} for resume_text)
        prompt = f'''act as an expert resume scanner to read {resume_text} and {jd_text} and 
                    give score,percentage and count for the numerical features 
                    {{
                        "skill_match_percentage": 75,
                        "critical_skill_match_percentage": 75,
                        "project_relevance_score": 75,
                        "certification_relevance_score": 75,
                        "internship_relevance_score": 75,
                        "resume_completeness_score": 75,
                        "keyword_match_score": 75,
                        "role_category_match_score": 75,
                        "missing_skills_count": 6,
                        "critical_missing_skills_count": 8,
                        "missing_skills": ["Docker", "SQL"],
                        "critical_missing_skills": ["Docker", "SQL"],
                        "matched_skills": ["Docker", "SQL"],
                        "feedback": "Your resume shows strong technical skills but lacks SQL",
                        "roadmap_7_day": "Complete SQL course and build 2 projects using it",
                        "roadmap_30_day": "Complete SQL course and build 2 projects using it",
                        "Resume_improvement_suggestion": "Complete SQL course and build 2 projects using it",
                        "Job-specific_preparation_suggestions": "Complete SQL course and build 2 projects using it"
                    }}
                    Analyze the actual resume and job description below, then return ONLY a JSON 
                    object in the following format, with your own real values replacing the examples'''
        
        # Step 2: call Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":prompt}],
            response_format={"type": "json_object"}
        )
        
        # Step 3: parse JSON
        data = json.loads(response.choices[0].message.content)
        
        # Step 4: fix count mismatches
        data["missing_skills_count"] = len(data["missing_skills"])
        data["critical_missing_skills_count"] = len(data["critical_missing_skills"])

        readiness_score = sum(data[col] for col in bucket_a_cols) / len(bucket_a_cols)
        data["placement_readiness_score"] = round(readiness_score, 2)
        
        # Step 5: build input row for the model
        input_row = pd.DataFrame([data])[feature_order]
        
        # Step 6: predict
        prediction = model.predict(input_row.values)
                
        
        # Step 7: add prediction into data
        data["predicted_readiness_label"] = prediction[0]
        
        # Step 8: return everything
        return data

def create_pdf(result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for key, value in result.items():
        text = f"{key}: {value}"
        text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    
    return bytes(pdf.output())

st.title('Placement Readiness Intelligence System')
uploaded_resume = st.file_uploader("Upload your resume in pdf format", type=["pdf"])
jd_input = st.text_area("Enter or paste the job description")
analyze_button = st.button("Analyze")

if analyze_button:
    if uploaded_resume is not None and jd_input.strip() != "":
        reader = pypdf.PdfReader(uploaded_resume)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()
        result = analyze_resume(resume_text, jd_input)
        st.write(result)
        print(result)
        pdf_bytes = create_pdf(result)
        print(len(pdf_bytes))
        st.download_button(
                    label="Download summary PDF",
                    data=pdf_bytes,
                    file_name="placement_readiness_report.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("Please upload a resume and enter a job description")

        