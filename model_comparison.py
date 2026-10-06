"""
Model comparison script for the CVD IA2 project.

Trains 5 standalone models plus the final Stacking Ensemble on the SAME
train/val/test split used by cvd.py (random_state=42), tunes each model's
own decision threshold on the validation set (maximizing F1, exactly like
cvd.py's choose_threshold), and reports test-set performance for all six
side by side. This is what justifies combining LightGBM + XGBoost + Random
Forest into a Stacking Classifier rather than shipping a single model.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================
# 1. Load data (identical split to cvd.py)
# ============================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_selected.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(DATA_PATH)
features = [c for c in data.columns if c not in ["SEQN", "CVD"]]
X, y = data[features], data["CVD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

pos, neg = y_tr.sum(), (y_tr == 0).sum()
ratio = neg / max(pos, 1)

# ============================
# 2. Define all six models
# ============================
lgbm = LGBMClassifier(
    n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, class_weight="balanced", verbose=-1,
)
xgb = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, random_state=42, eval_metric="logloss",
    tree_method="hist", scale_pos_weight=ratio,
)
rf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42,
    class_weight="balanced",
)
logreg = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
dtree = DecisionTreeClassifier(
    max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42,
)
stack = StackingClassifier(
    estimators=[("lgbm", lgbm), ("xgb", xgb), ("rf", rf)],
    final_estimator=LogisticRegression(max_iter=200),
    stack_method="predict_proba", n_jobs=-1,
)

models = {
    "Logistic Regression": logreg,
    "Decision Tree": dtree,
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgbm,
    "Stacking Ensemble (final)": stack,
}

# ============================
# 3. Helpers (same F1-threshold logic as cvd.py)
# ============================
def choose_threshold_f1(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.nanargmax(f1[:-1])])

def evaluate(model, X_te, y_te, thr):
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= thr).astype(int)
    return {
        "AUROC": roc_auc_score(y_te, p),
        "AUPRC": average_precision_score(y_te, p),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "Accuracy": accuracy_score(y_te, pred),
        "Threshold": thr,
        "ConfMatrix": confusion_matrix(y_te, pred),
    }

# ============================
# 4. Train + evaluate every model
# ============================
results = {}
for name, model in models.items():
    print(f"[INFO] Training {name}...")
    model.fit(X_tr, y_tr)
    p_val = model.predict_proba(X_val)[:, 1]
    thr = choose_threshold_f1(y_val, p_val)
    res = evaluate(model, X_test, y_test, thr)
    results[name] = res
    print(f"  -> threshold={thr:.3f}  AUROC={res['AUROC']:.3f}  AUPRC={res['AUPRC']:.3f}  "
          f"F1={res['F1']:.3f}  Precision={res['Precision']:.3f}  Recall={res['Recall']:.3f}")

# ============================
# 5. Save comparison table
# ============================
rows = []
for name, res in results.items():
    rows.append({
        "Model": name,
        "AUROC": round(res["AUROC"], 3),
        "AUPRC": round(res["AUPRC"], 3),
        "Precision": round(res["Precision"], 3),
        "Recall": round(res["Recall"], 3),
        "F1": round(res["F1"], 3),
        "Accuracy": round(res["Accuracy"], 3),
        "Threshold": round(res["Threshold"], 3),
    })
df_results = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
df_results.insert(0, "Rank", range(1, len(df_results) + 1))
df_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("\n[OK] Comparison table saved to results/model_comparison.csv")
print(df_results.to_string(index=False))

# ============================
# 6. Chart: AUROC and F1 side by side
# ============================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
order = df_results["Model"].tolist()
colors = ["#C0504D" if "final" in m else "#4472C4" for m in order]

axes[0].barh(order, df_results.set_index("Model").loc[order, "AUROC"], color=colors)
axes[0].set_xlabel("AUROC (test set)")
axes[0].set_xlim(0, 1)
axes[0].invert_yaxis()
axes[0].set_title("Test AUROC by Model")
for i, v in enumerate(df_results.set_index("Model").loc[order, "AUROC"]):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

axes[1].barh(order, df_results.set_index("Model").loc[order, "F1"], color=colors)
axes[1].set_xlabel("F1-score, CVD=1 (test set, tuned threshold)")
axes[1].set_xlim(0, max(df_results["F1"]) + 0.1)
axes[1].invert_yaxis()
axes[1].set_title("Test F1 (positive class) by Model")
for i, v in enumerate(df_results.set_index("Model").loc[order, "F1"]):
    axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison.png", dpi=130)
plt.close()
print("[OK] Chart saved to figures/model_comparison.png")

# ============================
# 7. Persist confusion matrices as text for the report
# ============================
with open(RESULTS_DIR / "model_comparison_confusion.txt", "w") as f:
    for name, res in results.items():
        f.write(f"=== {name} (threshold={res['Threshold']:.3f}) ===\n")
        f.write(str(res["ConfMatrix"]))
        f.write("\n\n")
print("[OK] Confusion matrices saved to results/model_comparison_confusion.txt")
