# =========================
# Employee Attrition Training Script
# =========================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# =========================
# 2. Drop useless columns
# =========================

df.drop(["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], axis=1, inplace=True)

# =========================
# 3. Encode Target
# =========================

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# =========================
# 4. Encode Categorical Columns
# =========================

label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# 5. Split Data
# =========================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Handle Imbalance using SMOTE
# =========================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# 7. Feature Scaling
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 8. Train Model
# =========================

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# =========================
# 9. Evaluation
# =========================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 10. Save Model
# =========================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("encoder.pkl", "wb"))

print("\nModel Saved Successfully!")