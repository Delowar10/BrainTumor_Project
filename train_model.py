import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# ধরো X, y already তৈরি (তোমার original code থেকে)

# =====================
# PREPROCESS
# =====================
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =====================
# MODEL TRAIN
# =====================
model = ExtraTreesClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# =====================
# SAVE EVERYTHING
# =====================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("Model saved successfully!")