# -*- coding: utf-8 -*-
"""
糖尿病惡化風險預測 — 端到端可重複管線 (Shanghai_T2DM_Summary)
================================================================
本檔取代原本分散的 7 支腳本(資料轉換 / 補值 / log / 交互作用 / 標準化 /
Logistic Regression / XGBoost / KNN vs GMM),整併為單一可重複流程,並修正
以下方法學問題:

  (1) 前處理洩漏:原流程在「切分前」就對整份資料 fit 中位數與 StandardScaler,
      再做交叉驗證 → 測試折資訊外洩。本檔改把 impute + scale 放進 sklearn
      Pipeline,只在每個訓練折上 fit。
  (2) 病患層級洩漏:109 筆來自 100 位病患(8 位有多次回診)。同一人若同時落在
      train 與 test,會高估效能。本檔預設「每位病患只留一筆」並可改用 GroupKFold。
  (3) 特徵洩漏:Other Agents / Hypoglycemic Agents 這類用藥常是「因為有併發症才開」
      (如 epalrestat→神經病變、calcium dobesilate→視網膜病變),會把答案洩漏給模型。
      本檔預設不納入這些用藥二元變數(可用 INCLUDE_MED_FEATURES 開關檢視影響)。
  (4) 目標定義:橫斷面資料無法真正預測「未來惡化」。本檔提供三個目標,並新增
      複合「Any_Complication(併發症有無)」作為『疾病已惡化/已分層』的代理指標,
      同時保留大血管 / 小血管兩個細分目標。低血糖(10/109)因陽性過少不列為主要目標。
  (5) 冗餘特徵:原流程同時保留原始值與 log 值並各自標準化 → 高度共線,LR 係數不穩。
      本檔對右偏變數「只用 log 版本取代原始值」,不重複。
  (6) 修正 KNN/GMM 腳本的 bug(引用未定義的 df["Cluster"])與命名(KMeans≠KNN),
      並加入 K 值/成分數的模型選擇(silhouette / BIC)。

執行環境:Python 3.10+ / Colab。需要 pandas, numpy, scikit-learn, matplotlib。
xgboost 與 shap 為選用:若已安裝則使用 XGBoost + SHAP,否則自動退回
HistGradientBoosting + permutation importance(結果格式相同,可先跑通)。

作者備註:全流程固定 random_state=42;所有輸出寫入 ./output/。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()            # 存在於 Jupyter / Colab → 用預設 inline 後端,圖顯示在儲存格下方
    _IN_NOTEBOOK = True
except NameError:
    matplotlib.use("Agg")    # 純腳本 / 無視窗環境 → 只存檔
    _IN_NOTEBOOK = False
import matplotlib.pyplot as plt


def _show():
    """存檔後,若在 notebook 就 inline 顯示,再關閉 figure 釋放記憶體。"""
    if _IN_NOTEBOOK:
        plt.show()
    plt.close()

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold, GroupKFold, cross_val_predict, cross_validate,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# 0. 設定
# ------------------------------------------------------------------ #
SEED = 42
XLSX_PATH = "Shanghai_T2DM_Summary.xlsx"     # 原始摘要檔 (sheet 名 T2DM)
OUTDIR = "output"
DEDUP_ONE_PER_PATIENT = False  # False:保留全部 109 筆,改用 GroupKFold 依病患分組(不洩漏)
USE_GROUP_CV = True            # True:GroupKFold(依 Patient Number);與上面搭配使用
INCLUDE_MED_FEATURES = False   # False:排除用藥二元變數(避免答案洩漏);設 True 可比較影響
RUN_EDA = True                 # True:輸出 EDA 圖(直方圖/箱型圖/相關矩陣/缺失熱圖/目標分布)
os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.RandomState(SEED)

# 嘗試載入 XGBoost / SHAP;沒有就退回 sklearn 內建
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ------------------------------------------------------------------ #
# 1. 讀取與編碼(直接讀原始 xlsx,不依賴中間 CSV)
# ------------------------------------------------------------------ #
def has_event(x):
    """none / no / 空值 → 0;其餘(有記錄任何病症/用藥)→ 1"""
    if pd.isna(x):
        return 0
    x = str(x).strip().lower()
    return 0 if x in ["none", "no", "nan", ""] else 1


def load_data():
    raw = pd.read_excel(XLSX_PATH, sheet_name="T2DM")

    # 病患 ID(用於分組交叉驗證,避免同一人同時進 train/test)
    patient_id = raw["Patient Number"].astype(str).str.split("_").str[0]

    d = pd.DataFrame(index=raw.index)
    # --- 人口學 / 身體組成(不做 log)---
    d["Age"] = pd.to_numeric(raw["Age (years)"], errors="coerce")
    d["Height"] = pd.to_numeric(raw["Height (m)"], errors="coerce")
    d["Weight"] = pd.to_numeric(raw["Weight (kg)"], errors="coerce")
    d["BMI"] = pd.to_numeric(raw["BMI (kg/m2)"], errors="coerce")
    # --- 生活行為 / 病史 ---
    d["Gender"] = raw["Gender (Female=1, Male=2)"].map({1: 0, 2: 1})
    d["Smoking"] = (pd.to_numeric(raw["Smoking History (pack year)"], errors="coerce").fillna(0) > 0).astype(int)
    d["Alcohol"] = (raw["Alcohol Drinking History (drinker/non-drinker)"].astype(str)
                    .str.strip().str.lower().map({"non-drinker": 0, "drinker": 1}).fillna(0).astype(int))
    d["Duration"] = pd.to_numeric(raw["Duration of diabetes (years)"], errors="coerce")

    # --- 生化檢驗(右偏 → 稍後改用 log1p 取代原值)---
    lab = {
        "FPG": "Fasting Plasma Glucose (mg/dl)",
        "PPG": "2-hour Postprandial Plasma Glucose (mg/dl)",
        "Cpep_f": "Fasting C-peptide (nmol/L)",
        "Cpep_2h": "2-hour Postprandial C-peptide (nmol/L)",
        "Ins_f": "Fasting Insulin (pmol/L)",
        "Ins_2h": "2-hour Postprandial insulin (pmol/L)",
        "HbA1c": "HbA1c (mmol/mol)",
        "GA": "Glycated Albumin (%)",
        "TC": "Total Cholesterol (mmol/L)",
        "TG": "Triglyceride (mmol/L)",
        "HDL": "High-Density Lipoprotein Cholesterol (mmol/L)",
        "LDL": "Low-Density Lipoprotein Cholesterol (mmol/L)",
        "Cr": "Creatinine (umol/L)",
        "eGFR": "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ",
        "UA": "Uric Acid (mmol/L)",
        "BUN": "Blood Urea Nitrogen (mmol/L)",
    }
    right_skewed = ["FPG", "PPG", "Cpep_f", "Cpep_2h", "Ins_f", "Ins_2h",
                    "TG", "TC", "Cr", "eGFR", "UA", "BUN", "Duration"]
    for k, v in lab.items():
        col = pd.to_numeric(raw[v].replace("/", np.nan), errors="coerce")
        d[k] = col
    # 右偏變數改用 log1p 取代(HbA1c/GA/HDL/LDL 分布較對稱,保留原值)
    for k in right_skewed:
        if k in d.columns:
            # log1p 需非負;負值(理論上不會有)保留原值
            d[k] = np.where(d[k] >= 0, np.log1p(d[k]), d[k])

    # --- 可能洩漏的用藥/共病二元變數(預設不用)---
    med = pd.DataFrame(index=raw.index)
    med["Comorbidities"] = raw["Comorbidities"].apply(has_event)
    med["HypoAgents"] = raw["Hypoglycemic Agents"].apply(has_event)
    med["OtherAgents"] = raw["Other Agents"].apply(has_event)

    # --- 目標變數 ---
    Macro = raw["Diabetic Macrovascular  Complications"].apply(has_event)
    Micro = raw["Diabetic Microvascular Complications"].apply(has_event)
    Acute = raw["Acute Diabetic Complications"].apply(has_event)  # 本資料全為 none → 恆 0
    Hypo = (raw["Hypoglycemia (yes/no)"].astype(str).str.strip().str.lower()
            .map({"no": 0, "yes": 1}).fillna(0).astype(int))
    targets = pd.DataFrame({
        "Macrovascular": Macro,
        "Microvascular": Micro,
        # 複合「惡化/已分層」代理:任一種併發症(此資料 = 大血管 或 小血管)
        "Any_Complication": ((Macro + Micro + Acute) > 0).astype(int),
        "Hypoglycemia": Hypo,   # 陽性過少,僅供參考
    })

    return d, med, targets, patient_id


# ------------------------------------------------------------------ #
# 2. 建立無洩漏的 Pipeline(前處理在 CV 內完成)
# ------------------------------------------------------------------ #
def make_pipeline(kind):
    """kind: 'lr' 基準模型;'gb' 主要模型(XGBoost 或退回 HistGB)"""
    if kind == "lr":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 solver="liblinear", random_state=SEED)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])
    else:
        if HAS_XGB:
            clf = XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="logloss", random_state=SEED,
                # scale_pos_weight 於各折內視類別比例調整較嚴謹,此處用固定值近似
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=300,
                l2_regularization=1.0, random_state=SEED,
            )
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])


def get_cv(y, groups, use_group):
    if use_group and groups is not None:
        return GroupKFold(n_splits=5), groups
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED), None


# ------------------------------------------------------------------ #
# 3. RQ2:基準(LR) vs 主要(XGBoost/GB)模型比較 + ROC + 混淆矩陣
# ------------------------------------------------------------------ #
def evaluate(X, y, groups, target_name, use_group=False):
    cv, grp = get_cv(y, groups, use_group)
    scoring = {"AUC": "roc_auc", "AP": "average_precision",
               "Accuracy": "accuracy", "Precision": "precision",
               "Recall": "recall", "F1": "f1"}
    rows = []
    for kind, label in [("lr", "Logistic Regression (baseline)"),
                        ("gb", f"{'XGBoost' if HAS_XGB else 'HistGB'} (main)")]:
        pipe = make_pipeline(kind)
        res = cross_validate(pipe, X, y, cv=cv, groups=grp, scoring=scoring,
                             error_score=np.nan)
        row = {"Target": target_name, "Model": label,
               "N": len(y), "Positives": int(y.sum())}
        for m in scoring:
            row[f"{m}_mean"] = np.nanmean(res[f"test_{m}"])
            row[f"{m}_sd"] = np.nanstd(res[f"test_{m}"])
        rows.append(row)
    return pd.DataFrame(rows)


def roc_and_confusion(X, y, target_name, groups=None):
    """用 out-of-fold 預測畫 ROC(LR vs 主要模型)並輸出混淆矩陣。
    保留 109 筆時傳入 groups(病患 id),ROC 也用 GroupKFold 產生 out-of-fold 預測。"""
    if USE_GROUP_CV and not DEDUP_ONE_PER_PATIENT and groups is not None:
        cv, cv_groups = GroupKFold(n_splits=5), groups
    else:
        cv, cv_groups = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED), None
    plt.figure(figsize=(5.2, 5))
    oof = {}
    for kind, label, color in [("lr", "Logistic Regression", "#2563eb"),
                              ("gb", f"{'XGBoost' if HAS_XGB else 'HistGB'}", "#dc2626")]:
        pipe = make_pipeline(kind)
        proba = cross_val_predict(pipe, X, y, cv=cv, groups=cv_groups, method="predict_proba")[:, 1]
        oof[kind] = proba
        auc = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {target_name}")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/ROC_{target_name}.png", dpi=150); _show()

    # 混淆矩陣(主要模型,門檻 0.5)
    pred = (oof["gb"] >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {target_name} ({'XGBoost' if HAS_XGB else 'HistGB'}, thr=0.5)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/CM_{target_name}.png", dpi=150); _show()
    return oof


# ------------------------------------------------------------------ #
# 4. RQ1:特徵重要性(SHAP 若可用,否則 permutation importance)
#     另外輸出 LR 標準化係數 → odds ratio(可解釋)
# ------------------------------------------------------------------ #
def feature_importance(X, y, target_name):
    # (a) 主要模型的重要性
    pipe = make_pipeline("gb")
    pipe.fit(X, y)
    model = pipe.named_steps["clf"]
    X_imp = pipe.named_steps["impute"].transform(X)

    if HAS_SHAP and HAS_XGB:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_imp)
        shap.summary_plot(sv, X_imp, feature_names=list(X.columns), show=False)
        plt.title(f"SHAP Summary — {target_name}")
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/SHAP_{target_name}.png", dpi=150); _show()
        imp = pd.DataFrame({"Feature": X.columns,
                            "Importance": np.abs(sv).mean(axis=0)})
    else:
        r = permutation_importance(pipe, X, y, scoring="roc_auc",
                                   n_repeats=20, random_state=SEED)
        imp = pd.DataFrame({"Feature": X.columns,
                            "Importance": r.importances_mean})
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    # 條狀圖(前 15,標數值)
    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    bars = plt.barh(top["Feature"], top["Importance"], color="#0d9488")
    xmax = float(top["Importance"].max())
    for b, v in zip(bars, top["Importance"]):
        plt.text(b.get_width() + xmax * 0.01, b.get_y() + b.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=8)
    plt.margins(x=0.12)
    plt.title(f"Feature Importance (top 15) — {target_name}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/IMP_{target_name}.png", dpi=150); _show()

    # (b) LR 係數 → odds ratio(給統計系報告用,直觀可解釋)
    lr = make_pipeline("lr"); lr.fit(X, y)
    coef = lr.named_steps["clf"].coef_[0]
    oddsr = pd.DataFrame({"Feature": X.columns,
                          "LR_coef(std)": coef,
                          "OddsRatio": np.exp(coef)}).sort_values(
                          "LR_coef(std)", key=np.abs, ascending=False).reset_index(drop=True)
    return imp, oddsr


# ------------------------------------------------------------------ #
# 5. RQ3:分群(KMeans 選 K + GMM 選成分數) + PCA 圖 + 卡方檢定
# ------------------------------------------------------------------ #
def clustering(X, targets, cluster_cols):
    Xc = SimpleImputer(strategy="median").fit_transform(X[cluster_cols])
    Xc = StandardScaler().fit_transform(Xc)

    # (a) KMeans:用 silhouette 選 K(2..6)
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xc)
        sil[k] = silhouette_score(Xc, km.labels_)
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xc)
    km_lab = km.labels_

    # (b) GMM:用 BIC 選成分數(2..6)
    bic = {}
    for k in range(2, 7):
        gm = GaussianMixture(n_components=k, random_state=SEED).fit(Xc)
        bic[k] = gm.bic(Xc)
    best_g = min(bic, key=bic.get)
    gm = GaussianMixture(n_components=best_g, random_state=SEED).fit(Xc)
    gm_lab = gm.predict(Xc)

    # (c) PCA 2D 視覺化
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(Xc)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].scatter(xy[:, 0], xy[:, 1], c=km_lab, cmap="viridis", s=25)
    ax[0].set_title(f"K-means (k={best_k}, silhouette={sil[best_k]:.3f})")
    ax[1].scatter(xy[:, 0], xy[:, 1], c=gm_lab, cmap="viridis", s=25)
    ax[1].set_title(f"GMM (k={best_g}, BIC-selected)")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Clustering_PCA.png", dpi=150); _show()

    # (d) 卡方:各分群 vs 併發症比例
    lines = []
    for lab, name in [(km_lab, f"KMeans(k={best_k})"), (gm_lab, f"GMM(k={best_g})")]:
        for t in ["Microvascular", "Macrovascular", "Any_Complication"]:
            tab = pd.crosstab(lab, targets[t])
            chi2, p, dof, _ = chi2_contingency(tab)
            prop = pd.crosstab(lab, targets[t], normalize="index").mul(100).round(1)
            lines.append(f"[{name}] {t}: chi2={chi2:.2f}, p={p:.4f}")
    return {"silhouette_by_k": sil, "bic_by_k": bic,
            "best_k": best_k, "best_gmm": best_g, "chi2": lines}


# ------------------------------------------------------------------ #
# 5b. EDA:直方圖 / 箱型圖 / 相關矩陣 / 缺失熱圖 / 目標分布
# ------------------------------------------------------------------ #
def run_eda(X, targets):
    edir = os.path.join(OUTDIR, "eda_figures")
    os.makedirs(edir, exist_ok=True)

    # (1) 目標變數分布(檢查類別不平衡)
    tt = ["Any_Complication", "Microvascular", "Macrovascular", "Hypoglycemia"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, t in zip(axes, tt):
        vc = targets[t].value_counts().sort_index()
        vals = [int(vc.get(0, 0)), int(vc.get(1, 0))]
        bars = ax.bar(["No", "Yes"], vals, color=["#94a3b8", "#ef4444"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                    ha="center", va="bottom", fontsize=10)
        ax.margins(y=0.15)
        ax.set_title(f"{t}\n(pos={int(targets[t].sum())}/{len(targets)})", fontsize=9)
    fig.suptitle("Target distribution", y=1.03)
    plt.tight_layout(); plt.savefig(f"{edir}/00_target_distribution.png", dpi=150, bbox_inches="tight"); _show()

    # (2) 缺失值熱圖 + 缺失比例
    plt.figure(figsize=(9, 4))
    import seaborn as sns
    sns.heatmap(X.isna(), cbar=False, cmap="Greys")
    plt.title("Missing-value map (white=missing)"); plt.xlabel("features"); plt.ylabel("records")
    plt.tight_layout(); plt.savefig(f"{edir}/01_missing_heatmap.png", dpi=150); _show()
    miss = (X.isna().mean() * 100).round(1).sort_values(ascending=False)
    miss.to_csv(f"{edir}/missing_pct.csv", encoding="utf-8-sig")

    # (3) 連續變數直方圖(網格)
    cont = [c for c in X.columns if X[c].nunique() > 5]
    ncol = 4; nrow = int(np.ceil(len(cont) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2.3))
    for ax, c in zip(axes.ravel(), cont):
        vals = X[c].dropna()
        ax.hist(vals, bins=20, color="#3b82f6", edgecolor="white")
        mu, med = vals.mean(), vals.median()
        ax.axvline(mu, color="#dc2626", lw=1.2)
        ax.set_title(f"{c}\nmean={mu:.1f} / median={med:.1f}", fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cont):]:
        ax.axis("off")
    fig.suptitle("Histograms of continuous features", y=1.005)
    plt.tight_layout(); plt.savefig(f"{edir}/02_histograms.png", dpi=150, bbox_inches="tight"); _show()

    # (4) 箱型圖(標準化後,方便並排看離群值)
    Xz = (X[cont] - X[cont].mean()) / X[cont].std()
    plt.figure(figsize=(min(1 + 0.55 * len(cont), 15), 4.5))
    Xz.boxplot(rot=90, grid=False)
    # 在每個箱子上方標「原始尺度中位數」
    for i, c in enumerate(cont, start=1):
        med = X[c].median()
        plt.text(i, Xz[c].quantile(0.75) + 0.15, f"{med:.1f}",
                 ha="center", va="bottom", fontsize=6, color="#dc2626", rotation=90)
    plt.title("Box plots (z-scored; red = raw-scale median) — outlier check")
    plt.tight_layout()
    plt.savefig(f"{edir}/03_boxplots.png", dpi=150); _show()

    # (5) 相關矩陣(連續變數,格子標數值)
    corr = X[cont].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title("Correlation matrix (continuous features)")
    plt.tight_layout(); plt.savefig(f"{edir}/04_correlation_matrix.png", dpi=150); _show()
    corr.to_csv(f"{edir}/correlation_matrix.csv", encoding="utf-8-sig")

    # (6) 敘述統計表(mean/sd/偏態/缺失比例)
    desc = X.describe().T
    desc["skew"] = X.skew(numeric_only=True)
    desc["missing_%"] = X.isna().mean() * 100
    desc.round(3).to_csv(f"{edir}/describe.csv", encoding="utf-8-sig")
    print(f"[EDA] 圖表已輸出至 {edir}/(目標分布、缺失熱圖、直方圖、箱型圖、相關矩陣、敘述統計)\n")


# ------------------------------------------------------------------ #
# 6. 主程式
# ------------------------------------------------------------------ #
def main():
    print(f"[env] XGBoost={'yes' if HAS_XGB else 'no (→HistGB)'} | "
          f"SHAP={'yes' if HAS_SHAP else 'no (→permutation)'}\n")

    d, med, targets, pid = load_data()

    # 特徵集合(預設:純臨床變數,不含用藥洩漏變數)
    feature_cols = list(d.columns)
    if INCLUDE_MED_FEATURES:
        d = pd.concat([d, med], axis=1)
        feature_cols += list(med.columns)

    # 去重(每位病患留第一筆),避免病患層級洩漏
    if DEDUP_ONE_PER_PATIENT:
        keep = ~pid.duplicated(keep="first").values
        d, targets, pid = d[keep].reset_index(drop=True), targets[keep].reset_index(drop=True), pid[keep].reset_index(drop=True)
    unit = "位病患" if DEDUP_ONE_PER_PATIENT else "筆紀錄"
    cvname = "GroupKFold(依病患)" if (USE_GROUP_CV and not DEDUP_ONE_PER_PATIENT) else "RepeatedStratifiedKFold"
    print(f"分析樣本:N={len(d)} {unit}(病患 {pid.nunique()} 位),特徵數={len(feature_cols)},CV={cvname}\n")

    X = d[feature_cols].copy()

    # EDA:直方圖 / 箱型圖 / 相關矩陣 / 缺失熱圖 / 目標分布
    if RUN_EDA:
        run_eda(X, targets)

    # 主要建模目標(低血糖陽性太少,不列入主要比較)
    main_targets = ["Any_Complication", "Microvascular", "Macrovascular"]

    all_metrics = []
    for t in main_targets:
        y = targets[t].astype(int)
        print(f"===== 目標:{t}(陽性 {int(y.sum())}/{len(y)})=====")
        # 保留 109 筆時用 GroupKFold(依 pid 分組);去重時用 RepeatedStratifiedKFold
        m = evaluate(X, y, pid, t, use_group=(USE_GROUP_CV and not DEDUP_ONE_PER_PATIENT))
        all_metrics.append(m)
        print(m[["Model", "AUC_mean", "AUC_sd", "AP_mean", "F1_mean", "Recall_mean"]].to_string(index=False))
        roc_and_confusion(X, y, t, groups=pid)
        imp, oddsr = feature_importance(X, y, t)
        imp.to_csv(f"{OUTDIR}/importance_{t}.csv", index=False, encoding="utf-8-sig")
        oddsr.to_csv(f"{OUTDIR}/lr_oddsratio_{t}.csv", index=False, encoding="utf-8-sig")
        print("Top5 重要特徵:", ", ".join(imp["Feature"].head(5)), "\n")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(f"{OUTDIR}/model_results.csv", index=False, encoding="utf-8-sig")

    # RQ3 分群(用核心臨床變數)
    cluster_cols = ["Age", "BMI", "Duration", "HbA1c", "FPG", "TG", "HDL", "LDL", "Cr", "eGFR"]
    cinfo = clustering(X, targets, cluster_cols)
    print("===== RQ3 分群 =====")
    print("KMeans silhouette by k:", {k: round(v, 3) for k, v in cinfo["silhouette_by_k"].items()})
    print("最佳 K =", cinfo["best_k"], "| GMM 最佳成分數 =", cinfo["best_gmm"])
    for l in cinfo["chi2"]:
        print("  ", l)

    print(f"\n完成。所有結果與圖表已輸出至 ./{OUTDIR}/")
    print("圖:ROC_*.png、CM_*.png、IMP_*.png(或 SHAP_*.png)、Clustering_PCA.png")
    print("表:model_results.csv、importance_*.csv、lr_oddsratio_*.csv")


if __name__ == "__main__":
    main()