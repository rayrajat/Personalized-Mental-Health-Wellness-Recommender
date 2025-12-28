import streamlit as st
import pandas as pd
import numpy as np
# Near the top of app/app.py, replace your current loading code with:

import joblib
import os

# Try these paths in order - one should work
possible_paths = [
    '../models/mood_classifier_pipeline.joblib',           # from app/ → root → models/
    '../../models/mood_classifier_pipeline.joblib',        # if deeper nesting
    'models/mood_classifier_pipeline.joblib',              # if running from root
    '/workspaces/Personalized-Mental-Health-Wellness-Recommender/models/mood_classifier_pipeline.joblib'  # absolute (last resort)
]

model_pipeline = None
for path in possible_paths:
    if os.path.exists(path):
        model_pipeline = joblib.load(path)
        print(f"Model loaded from: {path}")
        break

if model_pipeline is None:
    raise FileNotFoundError("Could not find model file in any expected location")

# ─── Import your model & functions ──────────────────────────────────────
# Make sure these imports match your project structure
#from src.data import *          # if you moved functions there
# or just copy-paste the necessary parts here for simplicity in MVP

# We assume these are available in the same file or imported:
# - model_pipeline  (your fitted pipeline)
# - rec_df          (recommendation DataFrame)
# - get_personalized_recommendations function
# - preprocessor, scaler (if using similarity)

# For demo simplicity, we'll assume everything is defined/imported above

# ─── Streamlit App ──────────────────────────────────────────────────────
st.title("Personalized Mental Health & Wellness Recommender")
st.markdown("**ML Zoomcamp Capstone Project** – Not a substitute for professional help")

st.header("Tell us a bit about yourself")

# ─── Input Form ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=40, value=21)
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])

with col2:
    course = st.text_input("Course/Program", "BCS")
    year_of_study = st.selectbox("Year of Study", ["year 1", "year 2", "year 3", "year 4"])
    cgpa = st.selectbox("CGPA Range", ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"])

# Create input DataFrame (must match training columns exactly)
input_data = pd.DataFrame([{
    'choose_your_gender': gender,
    'age': age,
    'what_is_your_course': course,
    'year_of_study': year_of_study,
    'cgpa': cgpa,
    'marital_status': marital_status
}])

# ─── Prediction & Recommendation Button ─────────────────────────────────
if st.button("Get my recommendations", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            # Predict mood
            predicted_mood = model_pipeline.predict(input_data)[0]
            
            st.success(f"Predicted mood category: **{predicted_mood.upper()}**")
            
            # Get recommendations
            recs = get_personalized_recommendations(
                input_data,
                predicted_mood,
                top_n=4
            )
            
            st.subheader("Recommended Wellness Activities")
            
            for _, row in recs.iterrows():
                with st.expander(f"**{row['title']}** ({row['duration_min']} min) – {row['category']}"):
                    st.write(row['description'])
                    st.caption(f"Score: {row['score']:.2f}")
            
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.info("Please make sure the model and preprocessor are properly loaded.")

# ─── Footer / Disclaimer ────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ This is an educational project demonstration. If you're experiencing mental health concerns, please contact a qualified professional or trusted support service.")