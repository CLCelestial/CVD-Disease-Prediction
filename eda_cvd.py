#!/usr/bin/env python3
"""
EDA + Cleaning + Feature-selection script for NHANES CVD project.

Saves:
 - data/cleaned_dataset.csv
 - data/cleaned_selected.csv  (top-k selected features + SEQN + CVD)
 - data/feature_importances.csv
 - figures/feature_importances.png
 - figures/class_distribution.png, missing_values.png, correlation_matrix.png

Usage:
    python eda_cvd.py path/to/nhanes_csv_folder
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------
# CONFIG
# -----------------------
# Folder holding the raw NHANES 2021-2023 CSVs listed in FILES below.
# Pass it as the first argument or set NHANES_DIR; defaults to data/raw/.
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NHANES_DIR", ROOT / "data" / "raw"))
OUT_DIR = ROOT / "data"       # cleaned CSVs + feature_importances.csv
FIG_DIR = ROOT / "figures"    # EDA plots
FILES = {
    "DEMO": "DEMO_L.csv",
    "BMX": "BMX_L.csv",
    "BPX": "BPXO_L.csv",
    "CBC": "CBC_L.csv",
    "GLU": "GLU_L.csv",
    "HDL": "HDL_L.csv",
    "HSCRP": "HSCRP_L.csv",
    "INS": "INS_L.csv",
    "MCQ": "MCQ_L.csv",
    "TCHOL": "TCHOL_L.csv",
}

# Feature selection params
TOP_K = 10                # choose top-k features by importance
REMOVE_CORRELATED = True  # whether to drop highly correlated features among top-k
CORR_THRESHOLD = 0.85     # correlation threshold to consider "highly correlated"

# -----------------------
# Helpers
# -----------------------
def load_csv(name):
    fpath = DATA_DIR / FILES[name]
    df = pd.read_csv(fpath)
    df.columns = [c.upper() for c in df.columns]
    return df

# -----------------------
# Load & merge (same logic as your script)
# -----------------------
print("[INFO] Loading CSV files...")
dfs = {name: load_csv(name) for name in FILES}

demo = dfs["DEMO"][["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1","INDFMPIR"]].copy()
demo.rename(columns={"RIDAGEYR":"age","RIAGENDR":"sex","RIDRETH1":"race","INDFMPIR":"pir"}, inplace=True)

bmx = dfs["BMX"][["SEQN","BMXBMI","BMXWAIST"]]
bpx = dfs["BPX"].copy()
sy_cols = [c for c in bpx.columns if c.startswith("BPXSY")]
di_cols = [c for c in bpx.columns if c.startswith("BPXDI")]
for colset in [sy_cols, di_cols]:
    for c in colset:
        bpx[c] = bpx[c].replace(0, np.nan)
if sy_cols:
    bpx["SBP_mean"] = bpx[sy_cols].mean(axis=1)
else:
    bpx["SBP_mean"] = np.nan
if di_cols:
    bpx["DBP_mean"] = bpx[di_cols].mean(axis=1)
else:
    bpx["DBP_mean"] = np.nan
bpx = bpx[["SEQN","SBP_mean","DBP_mean"]]

glu = dfs["GLU"][["SEQN","LBXGLU"]].rename(columns={"LBXGLU":"GLU"})
hdl = dfs["HDL"][["SEQN", "LBDHDD"]].rename(columns={"LBDHDD":"HDL"}) if "LBDHDD" in dfs["HDL"].columns else dfs["HDL"][["SEQN","LBDHDL"]].rename(columns={"LBDHDL":"HDL"})
tc  = dfs["TCHOL"][["SEQN","LBXTC"]].rename(columns={"LBXTC":"TCHOL"})
crp = dfs["HSCRP"][["SEQN","LBXHSCRP"]].rename(columns={"LBXHSCRP":"HSCRP"})

ins_df = dfs["INS"]
ins_col = None
for candidate in ["LBXINSI","LBXIN"]:
    if candidate in ins_df.columns:
        ins_col = candidate
        break
if ins_col is None:
    ins = pd.DataFrame({"SEQN": ins_df["SEQN"], "INS": np.nan})
else:
    ins = ins_df[["SEQN", ins_col]].rename(columns={ins_col: "INS"})

cbc_cols = ["SEQN","LBXHGB","LBXHCT","LBXWBCSI","LBXPLTSI","LBXMCVSI","LBXRDW"]
cbc = dfs["CBC"][ [c for c in cbc_cols if c in dfs["CBC"].columns] ].copy()

# Label from MCQ
mcq = dfs["MCQ"].copy()
y_label = np.zeros(len(mcq), dtype=int)
for q in ["MCQ160B","MCQ160C","MCQ160E","MCQ160F"]:
    if q in mcq.columns:
        y_label = np.where(mcq[q] == 1, 1, y_label)
label = pd.DataFrame({"SEQN": mcq["SEQN"], "CVD": y_label})

# Merge
data = demo.merge(bmx, on="SEQN", how="left")\
           .merge(bpx, on="SEQN", how="left")\
           .merge(glu, on="SEQN", how="left")\
           .merge(hdl, on="SEQN", how="left")\
           .merge(tc,  on="SEQN", how="left")\
           .merge(crp, on="SEQN", how="left")\
           .merge(ins, on="SEQN", how="left")\
           .merge(cbc, on="SEQN", how="left")\
           .merge(label, on="SEQN", how="left")

data = data.dropna(subset=["CVD"])  # keep only labeled rows
print(f"[INFO] Raw merged dataset shape: {data.shape}")

# -----------------------
# EDA plots (class distribution, missingness, correlation)
# -----------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# class distribution
class_counts = data["CVD"].value_counts()
plt.figure(figsize=(5,4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="pastel")
plt.xticks([0,1], ["No CVD (0)","CVD (1)"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png")
plt.close()

# missingness
missing = data.isnull().mean().sort_values(ascending=False)
plt.figure(figsize=(8,4))
missing.head(20).plot(kind="bar")
plt.ylabel("% Missing")
plt.title("Top Missing Features")
plt.tight_layout()
plt.savefig(FIG_DIR / "missing_values.png")
plt.close()

# correlation heatmap (numeric)
numeric_cols = data.select_dtypes(include=[np.number]).columns.drop(["SEQN","CVD"])
plt.figure(figsize=(10,9))
corr = data[numeric_cols.tolist() + ["CVD"]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_matrix.png")
plt.close()

print("[INFO] EDA plots saved to:", FIG_DIR)

# -----------------------
# CLEANING & IMPUTATION
# -----------------------
# drop columns with 100% missing
cols_drop = missing[missing == 1.0].index.tolist()
if cols_drop:
    data = data.drop(columns=cols_drop)
    print("[INFO] Dropped 100% missing columns:", cols_drop)

# impute numeric cols by median
numeric_cols = data.select_dtypes(include=[np.number]).columns.drop(["SEQN","CVD"])
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# remove duplicates by SEQN
before = len(data)
data = data.drop_duplicates(subset=["SEQN"])
after = len(data)
print(f"[INFO] Removed {before-after} duplicate rows (if any).")

# save cleaned dataset
clean_path = OUT_DIR / "cleaned_dataset.csv"
data.to_csv(clean_path, index=False)
print("[OK] Cleaned dataset saved to:", clean_path)
print("Final cleaned shape:", data.shape)

# -----------------------
# FEATURE IMPORTANCE (RandomForest)
# -----------------------
print("\n[INFO] Running feature-importance based selection (RandomForest)...")

# prepare X,y for RF (drop SEQN and CVD)
X = data.drop(columns=["SEQN","CVD"])
y = data["CVD"].astype(int)

# Preserve column order
cols = X.columns.tolist()

# If any categorical columns (sex/race) are non-numeric, convert
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].astype("category").cat.codes

# simple split to fit importance model (no hyperparam tuning)
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)

# get importances
imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
imp.to_csv(OUT_DIR / "feature_importances.csv")
print("[OK] Feature importances saved to:", OUT_DIR / "feature_importances.csv")

# plot
plt.figure(figsize=(8,6))
imp.head(30).plot(kind="bar")
plt.title("Feature importances (RandomForest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importances.png")
plt.close()
print("[OK] Feature importance plot saved to:", FIG_DIR / "feature_importances.png")

# -----------------------
# SELECT TOP-K FEATURES
# -----------------------
top_features = imp.head(TOP_K).index.tolist()
print(f"\n[INFO] Top-{TOP_K} features by importance:\n", top_features)

# Optionally remove highly-correlated features among selected set
if REMOVE_CORRELATED:
    sub = data[top_features].copy()
    corr_sub = sub.corr().abs()
    to_remove = set()
    for i, feat in enumerate(corr_sub.columns):
        if feat in to_remove:
            continue
        for j in range(i+1, corr_sub.shape[1]):
            other = corr_sub.columns[j]
            if other in to_remove:
                continue
            if corr_sub.loc[feat, other] >= CORR_THRESHOLD:
                # drop the one with lower importance
                if imp[feat] >= imp[other]:
                    to_remove.add(other)
                else:
                    to_remove.add(feat)
    final_features = [f for f in top_features if f not in to_remove]
    if to_remove:
        print("[INFO] Removed correlated features among top-k:", sorted(list(to_remove)))
    else:
        print("[INFO] No highly correlated features removed among top-k.")
else:
    final_features = top_features

print("[INFO] Final selected features:", final_features)

# create selected dataset (keep SEQN, CVD)
selected_df = data[ ["SEQN","CVD"] + final_features ].copy()
selected_path = OUT_DIR / "cleaned_selected.csv"
selected_df.to_csv(selected_path, index=False)
print("[OK] Selected dataset saved to:", selected_path)

# -----------------------
# OPTIONAL: quick sanity stats on selected features
# -----------------------
print("\n[SUMMARY] Selected features median by class (CVD=0 vs 1):")
print(selected_df.groupby("CVD")[final_features].median().T)

print("\n[INFO] Feature-selection completed. Files written to:", OUT_DIR)
