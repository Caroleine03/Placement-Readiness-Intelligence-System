# Placement Readiness Intelligence System (PRIS)

PRIS is an AI-powered platform that analyzes a candidate's resume against a specific 
job description to assess placement readiness. The system identifies matched and 
missing skills, calculates relevance scores across multiple dimensions, and generates 
a placement readiness score with a readiness label. It also provides personalized 
feedback and an improvement roadmap to help candidates strengthen their skills before applying.

## Features

- [x] Resume PDF upload and text extraction
- [x] Job description input
- [x] LLM-powered (Groq/Llama 3.3) skill extraction and gap analysis
- [x] ML-based (Logistic Regression) placement readiness classification
- [x] Placement readiness score (0-100)
- [x] Matched / missing / critical missing skills
- [x] Personalized feedback and 7-day / 30-day improvement roadmap
- [x] Downloadable PDF summary report

## How It Works

1. The user uploads a resume PDF and pastes a job description into the Streamlit dashboard, 
   then clicks "Analyze."

2. The extracted resume text and job description are sent to Groq's LLM (Llama 3.3, via API).
    The Groq's LLM model is an existing, pretained model that is called remotely and it 
    performs the resume analysis, skill extraction and skill gap comparision that returns
    sturctured scores and skill lists as JSON

3. Here Logistic Regression model is used which is trained on a synthetic dataset of 2000 rows.
    The Logistic Rgeression model takes numeric outputs from 10 features and predicts the 
    final readiness label.

4. The system combines both outputs into a placement readiness score, matched/missing skills, 
   personalized feedback, and a 7-day/30-day improvement roadmap.

5. The full summary can be downloaded as a PDF report for future reference.

## Tech Stack

- **Frontend/Dashboard:** Streamlit
- **LLM:** Groq API (llama-3.3-70b-versatile)
- **ML Model:** scikit-learn (Logistic Regression)
- **PDF Processing:** pypdf (reading), fpdf2 (report generation)
- **Data:** pandas, numpy

## Dataset

A synthetic dataset was used instead of the provided Kaggle "Student Placement Prediction Dataset," 
since the Kaggle dataset does not include any job description text and therefore cannot support 
JD-comparison features (e.g., skill_match_percentage, keyword_match_score). A custom dataset of 2000 
rows was generated instead, with 10 engineered features (8 percentage/score-type and 2 count-type) 
and 4 balanced readiness_label classes (Highly Ready, Moderately Ready, Needs Improvement, Not Ready Yet). 
The Logistic Regression model was trained on this synthetic dataset.

## Model Performance

         precision    recall  f1-score   support

     Highly Ready       0.90      0.86      0.88        95
 Moderately Ready       0.76      0.93      0.83       100
Needs Improvement       0.91      0.77      0.83       111
    Not Ready Yet       0.87      0.84      0.85        94

         accuracy                           0.85       400
        macro avg       0.86      0.85      0.85       400
     weighted avg       0.86      0.85      0.85       400


## Setup Instructions

1. Clone the repository:
```bash
   git clone [your-repo-url]
   cd [repo-folder]
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API key:
```
    GROQ_API_KEY=your_key_here
```

4. Run the app:
```bash
   python -m streamlit run app.py
```

## Usage

1. Upload a resume PDF
2. Paste the job description
3. Click "Analyze"
4. Review your readiness score, matched/missing skills, feedback, and roadmap
5. Download the PDF summary report

## Known Limitations

- **Synthetic data boundaries:** The training data was generated using non-overlapping 
  numeric ranges per readiness label, by design. This means the reported 0.85 accuracy reflects 
  performance on data with clean class separation; real-world resumes may present more ambiguous 
  or overlapping feature combinations than the synthetic training data captures.

- **LLM output consistency:** Groq's LLM output generally follows the requested JSON schema, but is 
  not 100% guaranteed to include every field on every call. During testing, a `KeyError: 'missing_skills'` 
  occurred when Groq's response omitted that field. Defensive `.get(...)` fallback handling was added 
  to prevent the pipeline from crashing when a field is occasionally missing.


## Project Deliverables

- [x] Python codebase
- [x] Streamlit dashboard
- [x] Deployed demo link
- [x] Documentation/README
- [x] Video walkthrough

