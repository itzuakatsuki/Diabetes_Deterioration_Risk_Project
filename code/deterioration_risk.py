"""
糖尿病惡化風險綜合評估
============================================================
把分散的分析整合成「可解讀的惡化風險」,分兩個時間尺度(避免主觀加權):

  長期惡化風險(慢性併發症)= 分類模型對「任一併發症」的預測機率
      - 以 Logistic Regression + GroupKFold(依病患)的 out-of-fold 機率為分數
        (每位病患的分數皆來自沒看過他的模型,較不高估)。
      - 依三分位切成 低 / 中 / 高。

  短期惡化風險(血糖失控)= 由每位病患自己的馬可夫轉移矩陣,推算「未來 k 步後
      落在高血糖(>180)或低血糖(<70)」的機率(P^k,起點為 InRange)。
      - 在假設每一步均代表 15 分鐘的前提下，4、16、96 步分別標記為 1、4、24 小時情境。
        由於目前未檢查實際 timestamp，這些數值應解讀為 k-step scenario score，
        不能直接視為校準後的實際時鐘時間風險。
      - 以第 96 步的出範圍機率作為 24-step 短期風險分數；不預設其必然已達穩態。

  綜合:以兩軸中位數切成四象限,對應不同臨床處置,供高齡友善決策 App 使用。

依賴:pandas, numpy, scikit-learn, matplotlib;需與 diabetes_deterioration_pipeline.py
同目錄(重用 load_data)。CGM 讀 .xls 需 xlrd,或先轉成 .xlsx。

----------------------------------------------------------------
【風險分數的正確解讀】
----------------------------------------------------------------
長期軸 ( 使用LogisticRegression(class_weight="balanced") ):
  P(Any_Complication | 當前臨床特徵) 是橫斷面的關聯/風險分層分數，
  不是未來 1/3/5 年併發症發生率。

短期軸:
  由個人 CGM 的 Markov 轉移機率推算，但其可信度依賴時間間隔是否正確、
  轉移矩陣是否有足夠觀測，以及 Markov 齊次性假設是否合理。

四象限:
  以樣本中位數切分，只能視為探索性分層，不是外部驗證的臨床門檻。
  「立即介入、慢性追蹤」等文字應視為介面示意，不能取代醫師判斷。
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython(); _NB = True
except NameError:
    matplotlib.use("Agg"); _NB = False
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# 中文字型設定:自動選用系統可用者(Windows 微軟正黑體;Linux/Mac 其他)
# 若圖上中文顯示為方框「□」,就是這裡沒選到字型——把你系統有的字型名加到最前面即可。
_available = {f.name for f in _fm.fontManager.ttflist}
for _f in ["Microsoft JhengHei", "Microsoft YaHei", "PingFang TC", "Heiti TC",
           "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False   # 讓負號正常顯示

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from diabetes_deterioration_pipeline import load_data

SEED = 42
SUMMARY_XLSX = "Shanghai_T2DM_Summary.xlsx"
CGM_DIR = "Shanghai_T2DM"          # 放 CGM 檔的資料夾
OUTDIR = "output_risk"
STEP_MIN = 15                       # CGM 取樣間隔(分)
HORIZONS = {"1h": 4, "4h": 16, "24h": 96}   # k 步 = 時界 / 15 分
os.makedirs(OUTDIR, exist_ok=True)


def _show():
    if _NB:
        plt.show()
    plt.close()


# ---------- 1. 長期風險:分類模型 out-of-fold 機率 ----------
def long_term_risk():
    d, med, targets, pid, _ = load_data()   # load_data 現多回傳 d_raw(EDA 用)
    raw = pd.read_excel(SUMMARY_XLSX, sheet_name="T2DM")
    record_id = raw["Patient Number"].astype(str).values      # 與 d 同列序
    y = targets["Any_Complication"].astype(int)
    X = d.copy()                                              # 用藥洩漏特徵已排除
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                              solver="liblinear", random_state=SEED))])
    prob = cross_val_predict(lr, X, y, cv=GroupKFold(5), groups=pid,
                             method="predict_proba")[:, 1]
    return pd.DataFrame({"record": record_id, "patient": pid.values,
                         "long_risk": prob, "has_complication": y.values})


# ---------- 2. 短期風險:每位病患的馬可夫 P^k ----------
def load_cgm(fp):
    df = pd.read_excel(fp)
    dc = [c for c in df.columns if str(c).strip().lower().startswith("date")][0]
    gc = [c for c in df.columns if "cgm" in str(c).lower()][0]
    s = df[[dc, gc]].copy(); s.columns = ["ts", "cgm"]
    s["ts"] = pd.to_datetime(s["ts"], errors="coerce")
    s["cgm"] = pd.to_numeric(s["cgm"], errors="coerce")
    return s.dropna().sort_values("ts").reset_index(drop=True)


def st3(v):
    return 0 if v < 70 else (1 if v <= 180 else 2)   # 0=Low 1=InRange 2=High


def transition_matrix(states, max_gap_steps=2, ts=None):
    T = np.zeros((3, 3))
    for i in range(1, len(states)):
        T[states[i - 1], states[i]] += 1
    P = np.zeros((3, 3))
    for i in range(3):
        s = T[i].sum()
        P[i] = T[i] / s if s > 0 else np.eye(3)[i]   # 未出現的狀態 → 維持自身
    return P


def short_term_risk():
    files = sorted(glob.glob(os.path.join(CGM_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(CGM_DIR, "*.xls")))
    rows = []
    for fp in files:
        rid = os.path.splitext(os.path.basename(fp))[0]
        s = load_cgm(fp)
        g = s["cgm"].values
        if len(g) < 20:
            continue
        states = [st3(v) for v in g]
        P = transition_matrix(states)
        rec = {"record": rid, "patient": rid.split("_")[0]}
        for lab, k in HORIZONS.items():
            Pk = np.linalg.matrix_power(P, k)
            rec[f"hypo_{lab}"] = Pk[1, 0] * 100    # 起點 InRange → Low
            rec[f"hyper_{lab}"] = Pk[1, 2] * 100   # 起點 InRange → High
        # 短期風險分數 = 24h 後出範圍機率(≈ 穩態);另存實測 TIR/TBR/TAR 對照
        rec["short_risk"] = rec["hypo_24h"] + rec["hyper_24h"]
        rec["TIR"] = np.mean((g >= 70) & (g <= 180)) * 100
        rec["TBR"] = np.mean(g < 70) * 100
        rec["TAR"] = np.mean(g > 180) * 100
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------- 3. 合成與二維風險圖 ----------
def main():
    L = long_term_risk()
    S = short_term_risk()
    M = L.merge(S, on=["record", "patient"], how="inner")
    print(f"合併後樣本:{len(M)} 筆(長期 {len(L)}、短期 {len(S)})")

    # 三分位分組
    M["long_group"] = pd.qcut(M["long_risk"], 3, labels=["低", "中", "高"])
    M["short_group"] = pd.qcut(M["short_risk"].rank(method="first"), 3, labels=["低", "中", "高"])

    # 四象限(以中位數切)
    lt, stv = M["long_risk"].median(), M["short_risk"].median()
    def quad(r):
        hi_l = r["long_risk"] >= lt
        hi_s = r["short_risk"] >= stv
        if hi_l and hi_s: return "A 立即介入(長高+短高)"
        if hi_l and not hi_s: return "B 慢性追蹤(長高+短低)"
        if not hi_l and hi_s: return "C 血糖波動注意(長低+短高)"
        return "D 常規追蹤(長低+短低)"
    M["quadrant"] = M.apply(quad, axis=1)

    # 每位病患風險表
    cols = ["record", "patient", "long_risk", "long_group",
            "hypo_1h", "hyper_1h", "hypo_4h", "hyper_4h", "hypo_24h", "hyper_24h",
            "short_risk", "short_group", "TIR", "quadrant", "has_complication"]
    M[cols].round(3).to_csv(f"{OUTDIR}/patient_risk_table.csv", index=False, encoding="utf-8-sig")

    # 二維風險圖
    colors = {"A 立即介入(長高+短高)": "#dc2626", "B 慢性追蹤(長高+短低)": "#f59e0b",
              "C 血糖波動注意(長低+短高)": "#3b82f6", "D 常規追蹤(長低+短低)": "#16a34a"}
    plt.figure(figsize=(7.2, 6))
    for q, c in colors.items():
        sub = M[M["quadrant"] == q]
        plt.scatter(sub["long_risk"] * 100, sub["short_risk"], s=40, alpha=0.75,
                    color=c, edgecolor="white", linewidth=0.5, label=q)
    plt.axvline(lt * 100, color="grey", ls="--", lw=1)
    plt.axhline(stv, color="grey", ls="--", lw=1)
    plt.xlabel("長期惡化風險:P(任一併發症) (%)")
    plt.ylabel("短期惡化風險:未來 24h 出範圍機率 (%)")
    plt.title("糖尿病惡化風險二維分層圖")
    plt.legend(fontsize=8, loc="upper left", framealpha=0.9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/risk_map_2d.png", dpi=150); _show()

    # 象限計數 + 各象限實際併發症比例(驗證分層有效性)
    tab = M.groupby("quadrant").agg(
        人數=("record", "size"),
        平均長期風險=("long_risk", lambda x: round(x.mean() * 100, 1)),
        平均短期風險=("short_risk", lambda x: round(x.mean(), 1)),
        實際併發症比例=("has_complication", lambda x: round(x.mean() * 100, 1)),
    ).reset_index()
    tab.to_csv(f"{OUTDIR}/quadrant_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== 四象限摘要 ===")
    print(tab.to_string(index=False))

    print(f"\n完成。輸出於 ./{OUTDIR}/")
    print("圖:risk_map_2d.png    表:patient_risk_table.csv、quadrant_summary.csv")
    # 示範:印出幾筆高風險病患
    top = M.sort_values(["long_risk", "short_risk"], ascending=False).head(5)
    print("\n最高綜合風險前 5 筆:")
    print(top[["record", "long_risk", "short_risk", "quadrant"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
