import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ============================
# 1. Load Data
# ============================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_selected.csv"
MODEL_PATH = ROOT / "models" / "cvd_model.joblib"
FIG_DIR = ROOT / "figures"
data = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded cleaned dataset: {data.shape}")

features = [c for c in data.columns if c not in ["SEQN", "CVD"]]
X, y = data[features], data["CVD"]

# Split → train / val / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Imbalance ratio for XGB weighting
pos, neg = y_tr.sum(), (y_tr == 0).sum()
ratio = neg / max(pos, 1)
print(f"[INFO] Training samples: pos={pos}, neg={neg}, ratio={ratio:.2f}")

# ============================
# 2. Define Model (Stacking)
# ============================
lgbm = LGBMClassifier(
    n_estimators=400, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, class_weight="balanced"
)

xgb = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss",
    tree_method="hist", scale_pos_weight=ratio
)

rf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    n_jobs=-1, random_state=42, class_weight="balanced"
)

stack = StackingClassifier(
    estimators=[("lgbm", lgbm), ("xgb", xgb), ("rf", rf)],
    final_estimator=LogisticRegression(max_iter=200),
    stack_method="predict_proba", n_jobs=-1
)

# ============================
# 3. Helper Functions
# ============================
def evaluate(model, name, X, y, thr):
    """Evaluate model performance at a given threshold."""
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    print(f"\n=== {name} @ threshold={thr:.3f} ===")
    print("AUROC:", roc_auc_score(y, p))
    print("AUPRC:", average_precision_score(y, p))
    print("Confusion Matrix:\n", confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=3))

def choose_threshold(y_true, proba, mode="f1", target=0.90):
    """Pick threshold by specificity or F1 or precision/recall constraints."""
    t_grid = np.linspace(0.001, 0.999, 999)
    if mode == "specificity":
        best_t, best_gap = 0.5, 1e9
        for t in t_grid:
            pred = (proba >= t).astype(int)
            tn = ((pred == 0) & (y_true == 0)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            spec = tn / max(tn + fp, 1)
            gap = abs(spec - target)
            if gap < best_gap:
                best_t, best_gap = t, gap
        return best_t
    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_true, proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[np.nanargmax(f1[:-1])])

# ============================
# 4. Train & Tune Threshold
# ============================
print("\n[INFO] Training STACK model...")
stack.fit(X_tr, y_tr)

# Tune for F1 (can switch to mode="specificity" if needed)
p_val = stack.predict_proba(X_val)[:, 1]
thr = choose_threshold(y_val, p_val, mode="f1")
print(f"[INFO] Chosen threshold (F1): {thr:.3f}")

# Evaluate
evaluate(stack, "TRAIN", X_tr, y_tr, thr)
evaluate(stack, "VALID", X_val, y_val, thr)
evaluate(stack, "TEST", X_test, y_test, thr)

# Save model + threshold for Flask app to use
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": stack, "features": features, "threshold": thr}, MODEL_PATH)
print(f"[OK] Model saved to {MODEL_PATH}")

# ============================
# 5. Threshold Curve Visualization (Optional)
# ============================
p_val = stack.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
best_idx = np.nanargmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label="Precision", color='blue')
plt.plot(thresholds, recalls[:-1], label="Recall", color='green')
plt.plot(thresholds, f1_scores[:-1], label="F1 Score", color='red')
plt.axvline(best_threshold, color='black', linestyle='--',
            label=f"Best F1 = {best_f1:.3f} @ thr={best_threshold:.3f}")
plt.title("Precision, Recall & F1 vs Threshold (Validation Set)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "threshold_curve.png")
plt.close()

print(f"[INFO] Best threshold for F1: {best_threshold:.3f} with F1={best_f1:.3f}")
