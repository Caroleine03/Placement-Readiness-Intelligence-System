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

model = joblib.load('random_forest_model.joblib') 

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
        prompt = f'''
                    You are an expert placement evaluator.

                    Analyze the candidate's resume against the job description carefully.

                    Your goal is to evaluate how well the candidate matches the specific
                    requirements of the job description.

                    Use the following scoring rules:

                    90-100 = Excellent alignment
                    80-89  = Strong alignment
                    70-79  = Moderate alignment
                    50-69  = Weak alignment
                    0-49   = Very weak alignment

                    Scoring criteria:

                    1. skill_match_percentage:
                    Compare the candidate's skills with the skills explicitly required
                    in the job description.

                    2. critical_skill_match_percentage:
                    Evaluate only the mandatory or critical skills required by the job.

                    3. project_relevance_score:
                    Evaluate how relevant the candidate's projects are to the target job.

                    4. certification_relevance_score:
                    Evaluate how relevant the candidate's certifications are to the target job.

                    5. internship_relevance_score:
                    Evaluate how relevant the candidate's internship or work experience
                    is to the target job.

                    6. resume_completeness_score:
                    Evaluate whether the resume contains important sections such as
                    education, skills, projects, experience and certifications.

                    7. keyword_match_score:
                    Compare important technical and role-specific keywords in the resume
                    with the job description.

                    8. role_category_match_score:
                    Evaluate how closely the candidate's overall background matches
                    the target job role.

                    Important consistency rules:

                    - Only consider skills relevant to the given job description.
                    - Do not give a high skill-match score when important required skills
                    are missing.
                    - Any skill identified as missing must appear in missing_skills.
                    - Any critical missing skill must appear in critical_missing_skills.
                    - missing_skills_count must equal the number of items in missing_skills.
                    - critical_missing_skills_count must equal the number of items in
                    critical_missing_skills.
                    - Do not say that a skill is missing in the feedback if it is not
                    included in missing_skills.
                    - Do not invent skills that are not relevant to the job description.
                    - Give realistic scores based on evidence from the resume and JD.

                    Return ONLY a JSON object in this format:

                    {{
                        "skill_match_percentage": 75,
                        "critical_skill_match_percentage": 70,
                        "project_relevance_score": 75,
                        "certification_relevance_score": 75,
                        "internship_relevance_score": 75,
                        "resume_completeness_score": 75,
                        "keyword_match_score": 70,
                        "role_category_match_score": 75,
                        "missing_skills_count": 2,
                        "critical_missing_skills_count": 1,
                        "missing_skills": ["Docker", "SQL"],
                        "critical_missing_skills": ["Docker"],
                        "matched_skills": ["Python", "Excel"],
                        "feedback": "Explain the candidate's strengths and important gaps.",
                        "roadmap_7_day": "Give a practical 7-day improvement plan.",
                        "roadmap_30_day": "Give a practical 30-day improvement plan.",
                        "Resume_improvement_suggestion": "Give specific resume improvement advice.",
                        "Job-specific_preparation_suggestions": "Give preparation advice specifically for this job."
                    }}

                    Resume:
                    {resume_text}

                    Job Description:
                    {jd_text}
                    '''
        
        # Step 2: call Groq
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
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
    pdf.set_auto_page_break(auto=True, margin=15)

    # Safe width for A4 page
    content_width = 170

    # Function to clean text for PDF
    def clean_text(text):
        text = str(text)
        text = text.replace("\\n", "\n")
        text = text.replace("\r", "")
        text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(
        content_width,
        12,
        "PRIS - Placement Readiness Report",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C"
    )

    pdf.ln(5)

    # --------------------------------------------------
    # PLACEMENT READINESS
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Placement Readiness",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    score = result.get("placement_readiness_score", 0)
    level = result.get("predicted_readiness_label", "Not Available")

    pdf.cell(
        content_width,
        8,
        clean_text(f"Readiness Score: {score} / 100"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.cell(
        content_width,
        8,
        clean_text(f"Readiness Level: {level}"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(4)

    # --------------------------------------------------
    # STRONG AREAS
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Strong Areas",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    matched_skills = result.get("matched_skills", [])

    if matched_skills:
        for skill in matched_skills:
            pdf.set_x(20)
            pdf.multi_cell(
                160,
                7,
                clean_text(f"- {skill}")
            )
    else:
        pdf.multi_cell(
            content_width,
            7,
            "No matched skills identified."
        )

    pdf.ln(3)

    # --------------------------------------------------
    # SKILLS TO IMPROVE
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Skills to Improve",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    missing_skills = result.get("missing_skills", [])

    if missing_skills:
        for skill in missing_skills:
            pdf.multi_cell(
                content_width,
                7,
                clean_text(f"- {skill}")
            )
    else:
        pdf.multi_cell(
            content_width,
            7,
            "No major missing skills were identified."
        )

    pdf.ln(3)

    # --------------------------------------------------
    # CRITICAL SKILL GAPS
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Critical Skill Gaps",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    critical_skills = result.get("critical_missing_skills", [])

    if critical_skills:
        for skill in critical_skills:
            pdf.multi_cell(
                content_width,
                7,
                clean_text(f"- {skill}")
            )
    else:
        pdf.multi_cell(
            content_width,
            7,
            "No critical skill gaps were identified."
        )

    pdf.ln(3)

    # --------------------------------------------------
    # EVALUATOR FEEDBACK
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Evaluator Feedback",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    feedback = result.get("feedback", "")

    if feedback:
        pdf.multi_cell(
            content_width,
            7,
            clean_text(feedback)
        )

    pdf.ln(3)

    # --------------------------------------------------
    # 7-DAY ROADMAP
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "7-Day Improvement Roadmap",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    roadmap_7_day = result.get("roadmap_7_day", "")

    if roadmap_7_day:
        pdf.multi_cell(
            content_width,
            7,
            clean_text(roadmap_7_day)
        )

    pdf.ln(3)

    # --------------------------------------------------
    # 30-DAY ROADMAP
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "30-Day Improvement Roadmap",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    roadmap_30_day = result.get("roadmap_30_day", "")

    if roadmap_30_day:
        pdf.multi_cell(
            content_width,
            7,
            clean_text(roadmap_30_day)
        )

    pdf.ln(3)

    # --------------------------------------------------
    # RESUME IMPROVEMENT
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Resume Improvement Suggestions",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    resume_suggestion = result.get(
        "Resume_improvement_suggestion",
        ""
    )

    if resume_suggestion:
        pdf.multi_cell(
            content_width,
            7,
            clean_text(resume_suggestion)
        )

    pdf.ln(3)

    # --------------------------------------------------
    # JOB-SPECIFIC PREPARATION
    # --------------------------------------------------

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        content_width,
        10,
        "Job-Specific Preparation",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", size=11)

    job_prep = result.get(
        "Job-specific_preparation_suggestions",
        ""
    )

    if job_prep:
        pdf.multi_cell(
            content_width,
            7,
            clean_text(job_prep)
        )

    # --------------------------------------------------
    # RETURN PDF
    # --------------------------------------------------

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

        # -------------------------------
        # PRIS Evaluator Output
        # -------------------------------

        st.header("👋 Hi! I'm your PRIS Placement Evaluator")

        st.subheader("🎯 Your Placement Readiness")

        score = result.get("placement_readiness_score", 0)
        level = result.get("predicted_readiness_label", "Not Available")

        st.metric(
            label="Placement Readiness Score",
            value=f"{score} / 100"
        )

        st.success(f"Readiness Level: **{level}**")
        st.subheader("💪 Your Strong Areas")
        matched_skills = result.get("matched_skills", [])

        if matched_skills:
            st.write("Skills that match the job requirements:")
            for skill in matched_skills:
                st.write(f"• **{skill}**")

        st.write(
            f"Your skill match is **{result.get('skill_match_percentage', 0)}%**, "
            f"and your project relevance score is **{result.get('project_relevance_score', 0)}%**."
        )

        st.write(
            f"Your certification relevance score is **{result.get('certification_relevance_score', 0)}%**, "
            f"which shows how well your certifications align with the job."
        )
        st.subheader("⚠️ Skills You Need to Improve")

        missing_skills = result.get("missing_skills", [])

        if missing_skills:
            st.write("Based on your resume and the job description, I found these missing skills:")

            for skill in missing_skills:
                st.write(f"• **{skill}**")
        else:
            st.success("Great! No major missing skills were identified.")

        st.subheader("🚨 Critical Skill Gaps")
        critical_skills = result.get("critical_missing_skills", [])
        if critical_skills:
                    st.warning(
                        "These skills are particularly important for the target job. "
                        "I recommend prioritizing them first."
                    )
        
                    for skill in critical_skills:
                        st.write(f"• **{skill}**")
        else:
            st.success("Excellent! No critical skill gaps were identified.")
        st.subheader("📝 Evaluator Feedback")

        feedback = result.get("feedback", "")

        if feedback:
            st.info(feedback)

        st.subheader("📅 Your 7-Day Improvement Roadmap")

        roadmap_7_day = result.get("roadmap_7_day", "")

        if roadmap_7_day:
            st.write(roadmap_7_day)

        st.subheader("🗓️ Your 30-Day Improvement Roadmap")

        roadmap_30_day = result.get("roadmap_30_day", "")

        if roadmap_30_day:
            st.write(roadmap_30_day)

        st.subheader("📄 Resume Improvement Suggestions")

        resume_suggestion = result.get("Resume_improvement_suggestion", "")

        if resume_suggestion:
            resume_suggestion = resume_suggestion.replace("\\n", "\n")
            st.info(resume_suggestion)

        st.subheader("🎯 Job-Specific Preparation")

        job_prep = result.get("Job-specific_preparation_suggestions", "")

        if job_prep:
            st.info(job_prep)

        
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

        