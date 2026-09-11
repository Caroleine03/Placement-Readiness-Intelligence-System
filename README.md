# Placement Readiness Intelligence System (PRIS)

PRIS (Placement Readiness Intelligence System) is an AI-powered placement evaluation platform that analyzes a candidate's resume against a specific job description.

The system combines **Large Language Model (LLM) based resume/JD analysis** with a **Machine Learning classification model** to evaluate placement readiness.

PRIS identifies matched and missing skills, generates multiple job-relevance features, calculates a placement readiness score, predicts a readiness category, and provides personalized recommendations to help candidates improve their employability.

---

## Features

- [x] Resume PDF upload and text extraction
- [x] Job description input
- [x] LLM-powered resume and job description analysis
- [x] Skill matching and skill-gap identification
- [x] Critical skill-gap identification
- [x] 10 engineered ML features
- [x] Placement readiness score (0-100)
- [x] ML-based readiness classification
- [x] Random Forest final prediction model
- [x] Matched skills display
- [x] Missing skills display
- [x] Critical skill gaps display
- [x] Personalized evaluator feedback
- [x] 7-day improvement roadmap
- [x] 30-day improvement roadmap
- [x] Resume improvement suggestions
- [x] Job-specific preparation suggestions
- [x] Downloadable PDF placement readiness report
- [x] Interactive Streamlit dashboard

---

## How It Works

PRIS follows a multi-stage pipeline:

### 1. Resume and Job Description Input

The candidate uploads their resume as a PDF and enters the target job description through the Streamlit dashboard.

The resume text is extracted using `pypdf`.

### 2. LLM-Based Resume and JD Analysis

The extracted resume text and job description are sent to the Groq API.

The LLM analyzes the candidate's profile against the specific job description and generates structured information including:

- Matched skills
- Missing skills
- Critical missing skills
- Skill match percentage
- Critical skill match percentage
- Project relevance score
- Certification relevance score
- Internship relevance score
- Resume completeness score
- Keyword match score
- Role category match score

The LLM returns these results in structured JSON format.

### 3. Feature Engineering

PRIS uses 10 numerical features for ML prediction.

#### Score-Based Features

1. `skill_match_percentage`
2. `critical_skill_match_percentage`
3. `project_relevance_score`
4. `certification_relevance_score`
5. `internship_relevance_score`
6. `resume_completeness_score`
7. `keyword_match_score`
8. `role_category_match_score`

#### Count-Based Features

9. `missing_skills_count`
10. `critical_missing_skills_count`

### 4. Placement Readiness Score

The placement readiness score is calculated using the average of the eight score-based features:

```text
Placement Readiness Score =
Sum of 8 score-based features / 8
```

The score is represented on a scale of 0 to 100.

### 5. Machine Learning Prediction

The 10 generated features are passed to the trained Machine Learning model.

Four classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Random Forest achieved the best overall performance and was selected as the final model.

The model predicts one of four readiness categories:

- Highly Ready
- Moderately Ready
- Needs Improvement
- Not Ready Yet

### 6. Personalized Evaluation

PRIS combines the ML prediction and LLM-generated analysis to present:

- Placement readiness score
- Readiness level
- Strong areas
- Matched skills
- Missing skills
- Critical skill gaps
- Evaluator feedback
- 7-day improvement roadmap
- 30-day improvement roadmap
- Resume improvement suggestions
- Job-specific preparation suggestions

### 7. PDF Report

The complete evaluation can be downloaded as a formatted PDF report for future reference.

---

## System Architecture

```text
Candidate Resume (PDF)
          |
          v
   PDF Text Extraction
        (pypdf)
          |
          |
Job Description --------+
                        |
                        v
                  Groq LLM
                  Analysis
                        |
                        v
              Structured JSON
                   Output
                        |
                        v
             Feature Engineering
                        |
                        v
              10 Numerical Features
                        |
                        v
               Random Forest
                   Model
                        |
                        v
            Readiness Classification
                        |
             +----------+----------+
             |                     |
             v                     v
    Placement Readiness      LLM Recommendations
         Score                      |
             |                      |
             +----------+-----------+
                        |
                        v
             PRIS Streamlit
                Dashboard
                        |
                        v
                  PDF Report
```

---

## Machine Learning Models

Four classification algorithms were evaluated to determine the most suitable model for PRIS.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.00% | 0.86 | 0.85 | 0.85 |
| Decision Tree | 77.25% | 0.77 | 0.77 | 0.77 |
| Random Forest | **89.50%** | **0.89** | **0.90** | **0.89** |
| Gradient Boosting | 89.00% | 0.89 | 0.89 | 0.89 |

### Final Model

**Random Forest** was selected as the final model because it achieved the highest accuracy and macro F1 score among the evaluated models.

Random Forest is an ensemble learning algorithm that combines multiple decision trees and can capture nonlinear relationships between the readiness features.

---

## Dataset

A synthetic dataset containing **2,000 records** was created for training the Machine Learning models.

The dataset contains:

- 10 numerical features
- 1 target variable
- 4 readiness categories

### Target Classes

- Highly Ready
- Moderately Ready
- Needs Improvement
- Not Ready Yet

The synthetic dataset was created because the PRIS system requires job-specific resume/JD comparison features such as:

- Skill match percentage
- Critical skill match percentage
- Keyword match score
- Role category match
- Missing skill counts

The available placement dataset did not contain job-description information required to generate these features.

Therefore, a custom synthetic dataset was created specifically for the PRIS ML pipeline.

---

## Model Training

The dataset was divided into:

- **80% training data**
- **20% testing data**

With 2,000 records:

```text
Training records : 1,600
Testing records  :   400
```

A fixed random state was used to make the train/test split reproducible.

The models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

---

## Placement Readiness Score

The readiness score is calculated from eight score-based features:

```text
skill_match_percentage
critical_skill_match_percentage
project_relevance_score
certification_relevance_score
internship_relevance_score
resume_completeness_score
keyword_match_score
role_category_match_score
```

The calculation is:

```text
Placement Readiness Score =
(
    skill_match_percentage
    + critical_skill_match_percentage
    + project_relevance_score
    + certification_relevance_score
    + internship_relevance_score
    + resume_completeness_score
    + keyword_match_score
    + role_category_match_score
) / 8
```

The resulting value is displayed on a scale of 0 to 100.

The Random Forest model separately uses all 10 features to predict the readiness category.

---

## Tech Stack

### Programming Language

- Python

### Frontend / Dashboard

- Streamlit

### LLM

- Groq API
- `openai/gpt-oss-120b`

### Machine Learning

- scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Data Processing

- pandas
- numpy

### Resume Processing

- pypdf

### Model Serialization

- joblib

### PDF Report Generation

- fpdf2

### Environment Management

- python-dotenv

---

## Project Structure

```text
PRIS/
│
├── app.py
├── notebook(1).ipynb
├── random_forest_model.joblib
├── requirements.txt
├── README.md
├── .env
│
└── student_data.csv
```

> **Note:** The `.env` file contains the API key and should never be uploaded to a public repository.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [your-repo-url]
cd [repo-folder]
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the `.env` File

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_key_here
```

Replace `your_key_here` with your Groq API key.

### 4. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

---

## Usage

1. Open the PRIS Streamlit application.
2. Upload a candidate resume in PDF format.
3. Paste the target job description.
4. Click **Analyze**.
5. PRIS extracts the resume text.
6. The LLM compares the resume with the job description.
7. The system generates the 10 ML features.
8. The placement readiness score is calculated.
9. Random Forest predicts the readiness category.
10. PRIS displays:
    - Placement readiness score
    - Readiness level
    - Matched skills
    - Missing skills
    - Critical skill gaps
    - Evaluator feedback
    - 7-day roadmap
    - 30-day roadmap
    - Resume improvement suggestions
    - Job-specific preparation
11. Download the complete evaluation as a PDF report.

---

## Example Output

```text
Placement Readiness Score: 93 / 100

Readiness Level: Highly Ready

Strong Areas:
- Python
- SQL
- Pandas
- NumPy
- Excel
- Power BI
- Statistics
- Data Visualization
- Machine Learning

Skills to Improve:
No major missing skills identified.

Critical Skill Gaps:
No critical skill gaps identified.
```

---

## Key Design Decisions

### Why Use an LLM?

Traditional Machine Learning models cannot directly understand raw resume text and job-description text.

The LLM is therefore used to:

- Understand resume content
- Understand job requirements
- Identify relevant skills
- Identify missing skills
- Generate structured numerical features
- Provide personalized recommendations

### Why Use Machine Learning?

The ML model provides a structured classification layer that maps the generated numerical features to a predefined readiness category.

This creates a hybrid architecture:

```text
LLM
 |
 v
Feature Generation
 |
 v
Machine Learning
 |
 v
Readiness Classification
```

### Why Random Forest?

Logistic Regression was initially used as a baseline because it is simple and interpretable.

Additional models were evaluated to determine whether more complex algorithms could improve classification performance.

Random Forest achieved the highest accuracy:

```text
89.50%
```

Therefore, Random Forest was selected as the final model.

---

## Known Limitations

### Synthetic Dataset

The Machine Learning model was trained using synthetic data rather than a large real-world placement dataset.

Therefore, the reported model performance should be interpreted as performance on the synthetic test set and should not be treated as real-world placement prediction accuracy.

### LLM Output Variability

The quality of the generated features depends on the LLM's interpretation of the resume and job description.

Structured JSON output and prompt-based consistency rules are used to improve reliability, but LLM-generated scores may still vary between analyses.

### Resume Text Extraction

PRIS uses `pypdf` for PDF text extraction.

Image-only or scanned resumes may not provide extractable text without an OCR solution.

### Synthetic Feature Relationships

The synthetic training data was designed using predefined feature ranges for the readiness categories.

This creates clearer class boundaries than may exist in real-world candidate data.

As a result, model performance on real-world data may differ.

### API Dependency

The LLM analysis requires access to the Groq API and a valid API key.

---

## Future Enhancements

Potential future improvements include:

- Training on a larger real-world placement dataset
- Adding OCR support for scanned resumes
- Adding model prediction probabilities
- Improving PDF visual design
- Adding interactive charts for feature scores
- Adding candidate history and progress tracking
- Adding more job-role categories
- Adding advanced resume parsing
- Adding cloud deployment
- Adding automated model retraining with new data
- Adding explainable ML techniques for prediction interpretation

---

## Project Deliverables

- [x] Python codebase
- [x] Jupyter Notebook
- [x] Streamlit dashboard
- [x] Random Forest trained model
- [x] LLM integration
- [x] PDF report generation
- [x] README documentation
- [x] Deployed/demo application

---

## Conclusion

PRIS combines **Generative AI and Machine Learning** to create a personalized placement-readiness evaluation system.

Instead of simply predicting whether a candidate is ready, the system provides an actionable analysis by identifying strengths, skill gaps, critical requirements, and personalized improvement plans.

The final system demonstrates how an LLM can be used for unstructured resume/JD understanding while a Machine Learning model provides structured readiness classification.
