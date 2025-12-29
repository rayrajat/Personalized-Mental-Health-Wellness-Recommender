# Personalized Mental Health & Wellness Recommender  
**ML Zoomcamp Capstone Project**

**Educational demonstration project**  
Predicts likely mood category of a student based on basic demographic and academic information  
and provides **personalized, non-clinical wellness activity recommendations**

> ⚠️ **Important legal & ethical disclaimer**  
> This application is **NOT** a medical or psychological diagnostic tool.  
> It is created strictly for educational purposes as part of ML Zoomcamp course.  
> **Never** use it instead of professional mental health support.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Goals & Scope](#project-goals--scope)
- [Dataset](#dataset)
- [Project Architecture & Flow](#project-architecture--flow)
- [Demo](#demo)
- [Technologies & Tools](#technologies--tools)
- [Project Structure](#project-structure)
- [Local Setup & Usage](#local-setup--usage)
- [Model Performance Summary](#model-performance-summary)
- [Interpretability & Fairness](#interpretability--fairness)
- [Recommendation System Design](#recommendation-system-design)
- [Evaluation Criteria Fulfillment (ML Zoomcamp)](#evaluation-criteria-fulfillment-ml-zoomcamp)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)
- [Future Work Ideas](#future-work-ideas)
- [Author & Acknowledgments](#author--acknowledgments)

## Problem Statement

Mental health challenges among university students have been rising significantly worldwide,  
particularly after the COVID-19 period. Many students experience:

- High levels of stress, anxiety and depressive symptoms
- Academic pressure and uncertainty about future
- Limited access to professional psychological support
- Stigma around seeking help
- Lack of simple, immediate, personalized coping strategies

**Business / Real-world problem we are trying to approach:**

> How can we create an **educational ML demonstration** that  
> 1. Recognizes potential negative mood patterns from easily available student profile information  
> 2. Offers safe, evidence-inspired, non-clinical wellness suggestions  
> 3. Remains completely transparent about its limitations and never pretends to be medical advice

This project **does NOT** aim to diagnose or treat anyone.  
It demonstrates how machine learning techniques can be applied responsibly to sensitive topics.

## Project Goals & Scope

**Main educational goal**  
Show complete end-to-end ML project following ML Zoomcamp methodology

**Concrete objectives achieved:**

- Understand & communicate mental health indicators from student survey data
- Build reliable tabular classification model (multi-class mood prediction)
- Create meaningful, safe, mood-aware recommendation system
- Provide strong interpretability of model decisions
- Develop production-like web interface (Streamlit)
- Create reproducible, well-documented project structure
- Demonstrate awareness of limitations, ethics and bias

**Explicitly out of scope:**
- Real clinical usage
- Collection/storage of real personal data
- Professional psychological advice
- Large-scale production deployment

## Dataset

**Source**  
[Student Mental Health](https://raw.githubusercontent.com/KadenShubert/student-mentalhealth-eda/main/Student_Mental_health.csv)  
**Rows**: ~101  
**Features**: 11 (gender, age, course, year of study, CGPA, marital status + 4 direct mental health questions)  
**Target (derived)**: 3-class mood category  
• positive/neutral  
• negative (reported depression/anxiety/panic attack)  
• seeking help (reported seeking specialist)

**Important dataset limitations**  
- Very small sample size  
- Self-reported answers (social desirability bias possible)  
- Specific cultural/academic context (mostly Malaysian students)  
- No temporal dimension  
- Severe class imbalance especially for "seeking help" group

## Project Architecture & Flow
User inputs → Streamlit interface
↓
Basic feature vector (age, gender, year, CGPA, marital status)
↓
Pre-trained XGBoost classifier → mood category prediction
↓
Hybrid recommendation engine
• Hard filter: mood-appropriate activities
• Soft boost: similarity to similar training users
• Diversity & business rules
↓
3–5 ranked wellness recommendations + explanations


## Demo
<video controls src="project_demo.mp4" title="Title"></video>


## Technologies & Tools

| Category                | Main technologies                              |
|-------------------------|------------------------------------------------|
| Data processing         | pandas, numpy                                  |
| Visualization           | matplotlib, seaborn                            |
| Modeling                | scikit-learn, xgboost, shap                    |
| Web application         | streamlit                                      |
| Model serialization     | joblib                                         |
| Containerization        | Docker (optional)                              |
| Python version          | 3.10+                                          |

## Project Structure

```text
.
├── app/                    # Streamlit application
│   └── app.py
├── data/
│   ├── raw/                # original downloaded file
│   └── processed/          # cleaned & feature-engineered data
├── models/                 # saved trained model(s)
├── notebooks/              # main exploration & modeling
│   └── 01_eda_modeling.ipynb
├── src/                    # reusable modules (optional)
│   ├── data.py
│   └── utils.py
├── screenshots/            # demo images
├── Dockerfile
├── requirements.txt
└── README.md

Local Setup & Usage
Bashgit clone https://github.com/rayrajat/Personalized-Mental-Health-Wellness-Recommender.git
cd YOUR_REPO_NAME

# Virtual environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Launch the application
streamlit run app/app.py

## Docker Deployment

You can run the application in a containerized environment:

```bash
# 1. Build the Docker image
docker build -t mental-health-recommender .

# 2. Run the container
docker run -p 8501:8501 mental-health-recommender

# Open in browser: http://localhost:8501

Model Performance Summary
Best model: XGBoost classifier
Main metric: Macro F1-score (due to strong class imbalance)
Validation strategy: 5-fold stratified cross-validation
Typical results range (depending on random state & final cleaning):


Macro F1      ~0.61–0.68
Negative F1   ~0.68–0.74
Seeking help F1  ~0.52–0.62 (hardest class)

Interpretability & Fairness
Main technique: SHAP TreeExplainer
Most influential features (typical ranking):

Marital status (Married → much higher probability of negative mood)
CGPA (lower ranges)
Age (younger students)
Year of study

Fairness checks performed:

Performance slicing by gender
Performance slicing by age groups
No extreme disparities observed (small sample limitation)

