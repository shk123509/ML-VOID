import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("early_prediction_500k.csv")

X = df.drop("future_glucose_360days", axis=1)
y = df["future_glucose_360days"]

num_cols = ["age", "bmi", "systolic_bp", "diastolic_bp", "current_glucose",
            "fasting_glucose", "hba1c", "physical_activity", "diet_quality"]

cat_cols = ["gender", "smoking", "alcohol", "family_history"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=4,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

joblib.dump(pipeline, "early_predict_model.pkl")

# -------------------------
# User Input
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

loaded = joblib.load("early_predict_model.pkl")
out = loaded.predict(user_df)[0]

print("Predicted Glucose After 360 Days:", round(out, 2))
