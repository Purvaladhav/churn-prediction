import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)
import os

os.makedirs("data", exist_ok=True)

# ── 1. Load Data & Models ─────────────────────────────────────
X_train, X_test, y_train, y_test = joblib.load("models/data_splits.pkl")
feature_names = joblib.load("models/feature_names.pkl")

model_names = ["logistic_regression", "random_forest", "xgboost"]
models = {name: joblib.load(f"models/{name}.pkl") for name in model_names}

# ── 2. ROC Curve — All Models ─────────────────────────────────
plt.figure(figsize=(8, 5))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1],[0,1],"k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("data/roc_curves.png", dpi=150)
plt.close()
print("Saved: data/roc_curves.png")

# ── 3. Confusion Matrix — Best Model ─────────────────────────
best_name = joblib.load("models/best_model_name.pkl")
best_model = models[best_name]
y_pred = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["No Churn", "Churned"],
    cmap="Blues", ax=ax
)
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150)
plt.close()
print("Saved: data/confusion_matrix.png")

# ── 4. Precision-Recall Curve ────────────────────────────────
y_prob_best = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_best)

plt.figure(figsize=(7, 4))
plt.plot(recall, precision, color="darkorange")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve — {best_name}")
plt.tight_layout()
plt.savefig("data/precision_recall.png", dpi=150)
plt.close()
print("Saved: data/precision_recall.png")

# ── 5. SHAP Feature Importance ───────────────────────────────
print("\nGenerating SHAP values (may take ~30 seconds)...")

xgb_model = models["xgboost"]
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("data/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/shap_summary.png")

# ── 6. Top 10 Features by SHAP ───────────────────────────────
mean_shap = np.abs(shap_values).mean(axis=0)
top_idx   = np.argsort(mean_shap)[::-1][:10]

plt.figure(figsize=(7, 5))
sns.barplot(x=mean_shap[top_idx],
            y=[feature_names[i] for i in top_idx],
            palette="viridis")
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 Features Driving Churn")
plt.tight_layout()
plt.savefig("data/shap_top10.png", dpi=150)
plt.close()
print("Saved: data/shap_top10.png")

print("\nAll evaluation plots saved to data/")
print(f"Best model: {best_name} — ready for deployment!")