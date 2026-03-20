import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import os

# ── Load Config ───────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

mc = cfg["models"]

# ── Load Data ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = joblib.load(cfg["paths"]["splits_path"])
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ── Define Models from Config ─────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(
        max_iter=mc["logistic_regression"]["max_iter"],
        random_state=mc["logistic_regression"]["random_state"]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=mc["random_forest"]["n_estimators"],
        random_state=mc["random_forest"]["random_state"]
    ),
    "xgboost": XGBClassifier(
        n_estimators=mc["xgboost"]["n_estimators"],
        max_depth=mc["xgboost"]["max_depth"],
        learning_rate=mc["xgboost"]["learning_rate"],
        eval_metric=mc["xgboost"]["eval_metric"],
        random_state=mc["xgboost"]["random_state"]
    )
}

# ── MLflow Experiment ─────────────────────────────────────────
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)

        # Log all params from config
        mlflow.log_params(mc[name])
        mlflow.log_param("model_type", name)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, name)

        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churned"]))
        print(f"AUC-ROC : {auc:.4f}  |  F1: {f1:.4f}")

    joblib.dump(model, f"{cfg['paths']['models_dir']}/{name}.pkl")
    results[name] = auc

# ── Leaderboard ───────────────────────────────────────────────
print("\n" + "="*40)
for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25} {auc:.4f}  {'█' * int(auc * 30)}")

best = max(results, key=results.get)
joblib.dump(best, cfg["evaluation"]["best_model_path"])
print(f"\nBest model: {best}")