import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import yaml
import os

# ── Load Config ───────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv(cfg["data"]["path"])

print("Shape:", df.shape)
print("\nChurn distribution:")
print(df[cfg["data"]["target_column"]].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# ── 2. Clean Data ─────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop(columns=["customerID"], inplace=True)
df[cfg["data"]["target_column"]] = df[cfg["data"]["target_column"]].map({"Yes": 1, "No": 0})

# ── 3. EDA Plot ───────────────────────────────────────────────
os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
plt.figure(figsize=(6, 4))
sns.countplot(x=cfg["data"]["target_column"], data=df,
              hue=cfg["data"]["target_column"], palette="Set2", legend=False)
plt.title("Churn Distribution")
plt.xticks([0, 1], ["No Churn", "Churned"])
plt.savefig(f"{cfg['paths']['data_dir']}/churn_distribution.png", bbox_inches="tight")
plt.close()
print("\nPlot saved.")

# ── 4. Encode Categorical Features ───────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ── 5. Split Features & Target ────────────────────────────────
X = df.drop(cfg["data"]["target_column"], axis=1)
y = df[cfg["data"]["target_column"]]

# ── 6. Scale Features ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 7. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"],
    stratify=y
)

# ── 8. SMOTE ──────────────────────────────────────────────────
if cfg["preprocessing"]["smote"]:
    sm = SMOTE(random_state=cfg["data"]["random_state"])
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train size: {X_train.shape[0]}")

# ── 9. Save ───────────────────────────────────────────────────
os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
joblib.dump(scaler,                       cfg["paths"]["scaler_path"])
joblib.dump((X_train, X_test, y_train, y_test), cfg["paths"]["splits_path"])
joblib.dump(X.columns.tolist(),           cfg["paths"]["feature_names_path"])
print("\nData preparation complete!")