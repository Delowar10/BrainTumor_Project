import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# =====================
# DUMMY DATA (replace with your dataset)
# =====================
# ⚠️ তুমি চাইলে তোমার original X, y use করবে
X = np.random.rand(200, 9)
y = np.random.randint(0, 2, 200)

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
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# =====================
# SAVE
# =====================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ Model saved successfully!")
