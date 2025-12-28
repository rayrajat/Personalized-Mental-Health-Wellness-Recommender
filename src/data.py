# src/data.py
import pandas as pd
import os

def load_raw_data():

    path = os.path.join("data", "raw", "student_mental_health.csv")
    return pd.read_csv(path)


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df.columns = (df.columns
                  .str.strip()
                  .str.replace('?', '')
                  .str.replace(' ', '_')
                  .str.lower())
    
    # Fix year_of_study inconsistencies
    year_map = {'year 1': 'year 1', 'Year 1': 'year 1', '1st year': 'year 1', ...}  # your mapping
    df['year_of_study'] = df['year_of_study'].str.lower().map(year_map).fillna(df['year_of_study'])
    
    # Add mood_category
    conditions = [
        (df['depression'] == 'Yes') | (df['anxiety'] == 'Yes') | (df['panic_attack'] == 'Yes'),
        (df['seek_treatment'] == 'Yes')
    ]
    df['mood_category'] = np.select(conditions, ['negative', 'seeking_help'], default='positive_neutral')
    
    return df


def get_feature_target(df: pd.DataFrame):
    """Split into features and target"""
    features = ['gender', 'age', 'course', 'year_of_study', 'cgpa', 'marital_status']
    X = df[features]
    y = df['mood_category']
    return X, y