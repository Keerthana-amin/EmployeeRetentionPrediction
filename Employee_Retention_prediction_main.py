# employee_main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from joblib import dump

# =========================
# Load dataset
# =========================
df = pd.read_excel(r"D:\Excel Files\DATA1.xlsx")

# Drop ID column
df.drop(columns=["enrollee_id"], inplace=True)

# =========================
# Replace ALL null values with 0
# =========================
df.fillna(0, inplace=True)

# =========================
# Encode categorical columns
# =========================
categorical_cols = df.select_dtypes(include="object").columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# =========================
# Split features & target

X = df.drop(columns=["target"])
y = df["target"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# =========================
# Train LightGBM model
# =========================
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Save model & encoders
# =========================
dump(model, "employee_lgbm_model.joblib")
dump(encoders, "employee_encoders.joblib")

print("✅ Model training completed and files saved")
