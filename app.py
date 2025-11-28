# -------------------------
# diabetes_predict.py
# -------------------------

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# File paths
# -------------------------
MODEL_FILE = "diabetes_model.pkl"
PIPELINE_FILE = "diabetes_pipeline.pkl"
DATA_FILE = "early_prediction_500k.csv"  # Your dataset

# -------------------------
# Columns
# -------------------------
NUM_COLS = ["age", "bmi", "systolic_bp", "diastolic_bp", 
            "current_glucose", "fasting_glucose", "hba1c", 
            "physical_activity", "diet_quality"]
CAT_COLS = ["gender", "smoking", "alcohol", "family_history"]

# -------------------------
# Pipeline function
# -------------------------
def build_pipeline(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    return full_pipeline

# -------------------------
# Train model if not exists
# -------------------------
if not os.path.exists(MODEL_FILE):
    print("Training model...")

    df = pd.read_csv(DATA_FILE)
    X = df.drop(["future_glucose_360days"], axis=1)
    y = df["future_glucose_360days"]

    pipeline = build_pipeline(NUM_COLS, CAT_COLS)
    X_prepared = pipeline.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=4,
        random_state=42
    )
    model.fit(X_prepared, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model trained and saved.")

else:
    # -------------------------
    # Load for inference
    # -------------------------
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # -------------------------
    # Get user input
    # -------------------------
    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    bmi = float(input("BMI: "))
    sbp = float(input("Systolic BP: "))
    dbp = float(input("Diastolic BP: "))
    cg = float(input("Current Glucose: "))
    fg = float(input("Fasting Glucose: "))
    hba = float(input("HbA1c: "))
    pa = int(input("Physical Activity (1-5): "))
    dq = int(input("Diet Quality (1-5): "))
    sm = input("Smoking (Yes/No): ")
    al = input("Alcohol (Yes/No): ")
    fh = input("Family History (Yes/No): ")

    user_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "current_glucose": cg,
        "fasting_glucose": fg,
        "hba1c": hba,
        "physical_activity": pa,
        "diet_quality": dq,
        "smoking": sm,
        "alcohol": al,
        "family_history": fh
    }])

    user_prepared = pipeline.transform(user_df)
    pred = model.predict(user_prepared)[0]

    print(f"\nPredicted Glucose After 360 Days: {round(pred, 2)}")
