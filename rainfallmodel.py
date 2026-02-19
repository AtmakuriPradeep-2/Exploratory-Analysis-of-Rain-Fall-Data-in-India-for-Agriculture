# =====================================
# Rainfall Prediction - Final Clean Code
# =====================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# =====================================
# 1️⃣ Load Dataset (SAFE VERSION)
# =====================================

data = pd.read_csv(
    "weatherAUS.csv",
    sep=None,
    engine="python",
    encoding="latin1"
)

# =====================================
# 2️⃣ Remove Rows Where Target Missing
# =====================================

data = data.dropna(subset=["RainTomorrow"])


# =====================================
# 3️⃣ Prepare Target & Features
# =====================================

y = data["RainTomorrow"].map({"No": 0, "Yes": 1}).astype(int)
X = data.drop("RainTomorrow", axis=1)

# Drop high-cardinality or problematic columns
if "Location" in X.columns:
    X = X.drop("Location", axis=1)

if "Date" in X.columns:
    X = X.drop("Date", axis=1)


# =====================================
# 4️⃣ Handle Missing Numeric Values
# =====================================

num_cols = X.select_dtypes(include=np.number).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())


# =====================================
# 5️⃣ Encode Categorical Columns
# =====================================

cat_cols = X.select_dtypes(include="object").columns

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])


# =====================================
# 6️⃣ Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 7️⃣ Initialize Models
# =====================================

XGBoost = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

Rand_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

svm = SVC()

Dtree = DecisionTreeClassifier(random_state=42)

GBM = GradientBoostingClassifier(random_state=42)

log = LogisticRegression(max_iter=1000)


# =====================================
# 8️⃣ Train Models
# =====================================

XGBoost.fit(X_train, y_train)
Rand_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)
Dtree.fit(X_train, y_train)
GBM.fit(X_train, y_train)
log.fit(X_train, y_train)


# =====================================
# 9️⃣ Compare Accuracy
# =====================================

models = {
    "XGBoost": XGBoost,
    "Random Forest": Rand_forest,
    "SVM": svm,
    "Decision Tree": Dtree,
    "Gradient Boosting": GBM,
    "Logistic Regression": log
}

print("\nModel Accuracy Comparison:\n")

for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")


# =====================================
# 🔟 Detailed Report (Best Model)
# =====================================

print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, XGBoost.predict(X_test)))
