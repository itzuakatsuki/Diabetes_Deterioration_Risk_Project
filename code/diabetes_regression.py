"""
迴歸問題:以臨床特徵預測 HbA1c(長期血糖控制)
====================================================
對應作業 3.5「迴歸問題:基準模型 Linear Regression + 主要模型 XGBoost」,
評估指標為 MAE、RMSE、MAPE、R²(作業 3.6)。與分類流程共用同一套資料處理
(直接 import diabetes_deterioration_pipeline.load_data),並同樣以 GroupKFold
依病患分組交叉驗證,避免病患層級洩漏。

目標可替換:改 TARGET 可預測其他連續變數(如 FPG、BMI)，但若更換 TARGET，
必須同步將目標欄位從特徵中排除，並檢查 load_data() 是否對該目標進行 log1p 轉換。
不能只修改 TARGET 一行 ; 
若要以 TIR 或 GMI 為目標，目前需另外撰寫病患編號合併流程；
本檔僅提供實作方向，尚未自動併入 cgm_metrics.csv。

環境:需要 pandas, numpy, scikit-learn, matplotlib;xgboost 選用(無則退回
HistGradientBoostingRegressor)。與 diabetes_deterioration_pipeline.py 放同目錄。

----------------------------------------------------------------
【X / Y 定義與解讀】
----------------------------------------------------------------
X:
  除 HbA1c 與 GA 外的臨床特徵。Patient Number 僅用於 GroupKFold 分組，
  不可作為預測特徵。

Y:
  該次臨床紀錄的 HbA1c 連續值。

本模型回答的是「目前臨床特徵能否估計同次量測的 HbA1c」，不是未來 HbA1c
變化的縱向預測。若要預測未來 HbA1c，需有明確基準日期、後續量測日期及時間順序。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()            # Jupyter / Colab / Spyder → inline 顯示
    _NB = True
except NameError:
    matplotlib.use("Agg")
    _NB = False
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

from diabetes_deterioration_pipeline import load_data   # 重用同一套資料處理

SEED = 42
OUTDIR = "output_reg"
os.makedirs(OUTDIR, exist_ok=True)

TARGET = "HbA1c"                       # 迴歸目標(連續變數);可改成 "FPG"、"BMI" 等
EXCLUDE_AS_FEATURE = ["HbA1c", "GA"]   # 排除目標與其近似指標(糖化白蛋白 GA 與 HbA1c r≈0.85)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _show():
    if _NB:
        plt.show()
    plt.close()


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return np.mean(np.abs((y[m] - p[m]) / y[m])) * 100


def make_reg(kind):
    """kind='lin' 基準(Linear Regression);'gb' 主要(XGBoost 或退回 HistGB)。"""
    if kind == "lin":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("reg", LinearRegression())])
    if HAS_XGB:
        reg = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED)
    else:
        reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                            max_iter=300, l2_regularization=1.0,
                                            random_state=SEED)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", reg)])


def main():
    print(f"[env] XGBoost={'yes' if HAS_XGB else 'no (→HistGB)'}")
    d, med, targets, pid, _ = load_data()   # load_data 現多回傳 d_raw(EDA 用)

    y = d[TARGET].astype(float)
    feats = [c for c in d.columns if c not in EXCLUDE_AS_FEATURE]
    X = d[feats].copy()
    keep = y.notna().values
    X, y, g = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), pid[keep].reset_index(drop=True)
    print(f"迴歸目標 = {TARGET} | N = {len(y)} | 特徵數 = {len(feats)} | CV = GroupKFold(依病患)\n")

    cv = GroupKFold(n_splits=5)
    rows, oof = [], {}
    main_label = f"{'XGBoost' if HAS_XGB else 'HistGB'} (main)"
    for kind, label, color in [("lin", "Linear Regression (baseline)", "#2563eb"),
                              ("gb", main_label, "#dc2626")]:
        pipe = make_reg(kind)
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=g)   # out-of-fold 預測
        oof[kind] = pred
        rows.append(dict(Model=label,
                         MAE=mean_absolute_error(y, pred),
                         RMSE=float(np.sqrt(mean_squared_error(y, pred))),
                         MAPE=mape(y, pred),
                         R2=r2_score(y, pred)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUTDIR}/reg_results.csv", index=False, encoding="utf-8-sig")
    print(R.round(3).to_string(index=False))

    # (1) 實際 vs 預測 散佈圖(兩模型)
    lo, hi = float(y.min()), float(y.max())
    plt.figure(figsize=(5.4, 5.2))
    plt.scatter(y, oof["lin"], s=22, alpha=0.6, color="#2563eb", label="Linear Regression")
    plt.scatter(y, oof["gb"], s=22, alpha=0.6, color="#dc2626", label=main_label.split()[0])
    plt.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="perfect")
    plt.xlabel(f"Actual {TARGET}"); plt.ylabel(f"Predicted {TARGET}")
    plt.title(f"Predicted vs Actual — {TARGET}")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/reg_pred_vs_actual_{TARGET}.png", dpi=150); _show()

    # (2) 殘差圖(主要模型)
    resid = y.values - oof["gb"]
    plt.figure(figsize=(6, 3.4))
    plt.scatter(oof["gb"], resid, s=20, alpha=0.6, color="#0d9488")
    plt.axhline(0, color="grey", ls="--", lw=1)
    plt.xlabel(f"Predicted {TARGET}"); plt.ylabel("Residual (actual − pred)")
    plt.title(f"Residuals — {main_label.split()[0]}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/reg_residuals_{TARGET}.png", dpi=150); _show()

    # (3) 特徵重要性(permutation,對主要模型)
    pipe = make_reg("gb"); pipe.fit(X, y)
    r = permutation_importance(pipe, X, y, scoring="r2", n_repeats=20, random_state=SEED)
    imp = pd.DataFrame({"Feature": X.columns, "Importance": r.importances_mean}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    imp.to_csv(f"{OUTDIR}/reg_importance_{TARGET}.csv", index=False, encoding="utf-8-sig")
    top = imp.head(12).iloc[::-1]
    plt.figure(figsize=(7, 4.5))
    bars = plt.barh(top["Feature"], top["Importance"], color="#7c3aed")
    xm = float(top["Importance"].max()) if top["Importance"].max() > 0 else 1
    for b, v in zip(bars, top["Importance"]):
        plt.text(b.get_width() + xm * 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", fontsize=8)
    plt.margins(x=0.14)
    plt.title(f"Permutation importance (top 12) — predicting {TARGET}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/reg_importance_{TARGET}.png", dpi=150); _show()

    print(f"\n完成。輸出於 ./{OUTDIR}/")
    print("圖:reg_pred_vs_actual_*.png、reg_residuals_*.png、reg_importance_*.png")
    print("表:reg_results.csv、reg_importance_*.csv")
    print(f"Top5 影響 {TARGET} 的特徵:", ", ".join(imp['Feature'].head(5)))


# -------------------------------------------------------------------- #
# 想改成預測 CGM 的 TIR / GMI(把血糖控制品質當迴歸目標)?
# 1) 先跑 cgm_lstm_markov.py 產生 cgm_output/cgm_metrics.csv
# 2) 用 Patient_Number 併回 load_data 的資料(病患編號一致),把 TIR/GMI 當 y
# 3) 設 TARGET="TIR"、EXCLUDE_AS_FEATURE=[]（TIR 不在臨床特徵內,無需排除）
# -------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
