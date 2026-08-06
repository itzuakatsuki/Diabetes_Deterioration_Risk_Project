"""
CGM 時間序列分析:血糖波動圖 + 馬可夫轉移矩陣 + LSTM 血糖預測
============================================================
輸入:Shanghai_T2DM/ 資料夾內 109 個 CGM 檔(每檔一位病患某次監測,每 15 分鐘一筆)。
本檔產出三塊,對應你們報告「LSTM-Markov 動態模型」的概念:

  1) CGM 指標(每筆監測):平均血糖、CV、GMI、TIR / TBR / TAR
     → 存成 cgm_metrics.csv,可用 Patient Number 併回 Shanghai_T2DM_Summary,
       當作額外特徵或分層依據(這是把 CGM 接進主模型的橋樑)。
  2) 血糖狀態馬可夫轉移矩陣(3 態與 5 態)+ 熱圖 + 長期穩態分布。
  3) LSTM 逐步血糖預測(用過去 N 筆預測下一筆)+ 實際 vs 預測「血糖波動圖」。
  4) 跨病患 pooled LSTM：將訓練病患的視窗合併訓練共同模型，
     並依病患分組交叉驗證，與 persistence baseline 及逐監測模型比較。

環境:Colab / Jupyter。需要 pandas, numpy, matplotlib, scikit-learn, tensorflow。
讀 .xls 需要 xlrd:  pip install xlrd tensorflow
(若你已先用 LibreOffice 把 .xls 轉成 .xlsx,就不需要 xlrd。)

固定 random_state=42;輸出寫入 ./cgm_output/。

----------------------------------------------------------------
【時間序列 X / Y 定義與限制】
----------------------------------------------------------------
LSTM 的 X:
  過去 look_back=12 筆 CGM（若每筆確為 15 分鐘，約為過去 3 小時）。

LSTM 的 Y:
  下一筆 CGM（horizon=1；若時間間隔有效，約為 15 分鐘後血糖）。

Markov:
  狀態轉移必須建立在真正相鄰的時間點上。若研究要宣稱「15 分鐘轉移」，
  正式分析應僅保留約 14–16 分鐘的時間差，並避免跨缺測區段與重複時間戳。

目前注意:
  markov_matrix() 的預設 max_gap_min=30，表示 30 分鐘內的間隔仍可能被計入；
  這不是嚴格的 15 分鐘轉移。報告需誠實註明，或後續改為 14–16 分鐘篩選。
"""

import os
import glob
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

SEED = 42
np.random.seed(SEED)

CGM_DIR = "Shanghai_T2DM"          # 放 109 個 CGM 檔的資料夾(.xls 或 .xlsx 皆可)
OUTDIR = "cgm_output"
os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------ #
# 1. CGM 讀取(自動處理 .xls 與 .xlsx)
# ------------------------------------------------------------------ #
def load_one(fp):
    """讀單一 CGM 檔,回傳 [ts, cgm] 已排序、去空值的 DataFrame。"""
    df = pd.read_excel(fp)   # pandas 會自動選 engine;.xls 需已安裝 xlrd
    date_col = [c for c in df.columns if str(c).strip().lower().startswith("date")][0]
    cgm_col = [c for c in df.columns if "cgm" in str(c).lower()][0]
    s = df[[date_col, cgm_col]].copy()
    s.columns = ["ts", "cgm"]
    s["ts"] = pd.to_datetime(s["ts"], errors="coerce")
    s["cgm"] = pd.to_numeric(s["cgm"], errors="coerce")
    return s.dropna().sort_values("ts").reset_index(drop=True)


def list_cgm_files():
    files = sorted(glob.glob(os.path.join(CGM_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(CGM_DIR, "*.xls")))
    if not files:
        raise FileNotFoundError(f"在 {CGM_DIR}/ 找不到 CGM 檔。請確認資料夾路徑。")
    return files


def patient_id_of(fp):
    return os.path.basename(fp).split("_")[0]


# ------------------------------------------------------------------ #
# 2. CGM 指標(TIR / TBR / TAR / GMI / CV)
#    國際共識目標範圍 70–180 mg/dL
# ------------------------------------------------------------------ #
def cgm_metrics(g):
    g = np.asarray(g, float)
    return dict(
        n=len(g),
        mean=g.mean(),
        SD=g.std(),
        CV=g.std() / g.mean() * 100,
        GMI=3.31 + 0.02392 * g.mean(),         # GMI(%) ← 平均血糖(mg/dL)
        TIR=np.mean((g >= 70) & (g <= 180)) * 100,   # Time In Range
        TBR=np.mean(g < 70) * 100,                   # Time Below Range
        TAR=np.mean(g > 180) * 100,                  # Time Above Range
    )


def build_metrics_table():
    rows = []
    for fp in list_cgm_files():
        s = load_one(fp)
        if len(s) < 10:
            continue
        m = cgm_metrics(s["cgm"].values)
        m["Patient_Number"] = os.path.splitext(os.path.basename(fp))[0]  # 例 2000_0_20201230
        m["patient"] = patient_id_of(fp)
        rows.append(m)
    met = pd.DataFrame(rows)
    met.to_csv(f"{OUTDIR}/cgm_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[metrics] {len(met)} 筆監測 → cgm_output/cgm_metrics.csv")
    print(met[["mean", "CV", "GMI", "TIR", "TBR", "TAR"]].describe().round(1).to_string())
    return met


# ------------------------------------------------------------------ #
# 3. 血糖狀態馬可夫轉移矩陣
# ------------------------------------------------------------------ #
STATE_DEFS = {
    3: (["Low(<70)", "InRange(70-180)", "High(>180)"],
        lambda v: 0 if v < 70 else (1 if v <= 180 else 2)),
    5: (["VLow(<54)", "Low(54-70)", "InRange(70-180)", "High(180-250)", "VHigh(>250)"],
        lambda v: 0 if v < 54 else (1 if v < 70 else (2 if v <= 180 else (3 if v <= 250 else 4)))),
}


def markov_matrix(n_states=3, max_gap_min=30):
    """
    以「相鄰兩筆 CGM」的狀態轉移建立母體轉移矩陣。
    max_gap_min:若兩筆間隔超過此分鐘數(資料中斷)則不計該轉移,避免跨斷點的假轉移。
    """
    labels, f = STATE_DEFS[n_states]
    T = np.zeros((n_states, n_states))
    for fp in list_cgm_files():
        s = load_one(fp)
        if len(s) < 10:
            continue
        st = s["cgm"].map(f).values
        gap = s["ts"].diff().dt.total_seconds().div(60).values  # 相鄰間隔(分)
        for i in range(1, len(st)):
            if np.isfinite(gap[i]) and gap[i] <= max_gap_min:
                T[st[i - 1], st[i]] += 1
    P = np.divide(T, T.sum(axis=1, keepdims=True),
                  out=np.zeros_like(T), where=T.sum(axis=1, keepdims=True) > 0)

    # 長期穩態分布(轉移矩陣的左特徵向量,特徵值=1)
    vals, vecs = np.linalg.eig(P.T)
    stat = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    stat = stat / stat.sum()

    # 熱圖
    fig, ax = plt.subplots(figsize=(1.6 * n_states + 1, 1.4 * n_states + 1))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_states)); ax.set_yticks(range(n_states))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("To state"); ax.set_ylabel("From state")
    ax.set_title(f"Glucose-state transition matrix ({n_states}-state)")
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    color="white" if P[i, j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/markov_{n_states}state.png", dpi=150); _show()

    pd.DataFrame(P, index=labels, columns=labels).to_csv(
        f"{OUTDIR}/markov_{n_states}state.csv", encoding="utf-8-sig")
    print(f"\n[markov] {n_states}-state 轉移矩陣 → cgm_output/markov_{n_states}state.png / .csv")
    print("  states:", labels)
    print("  P=\n", np.round(P, 3))
    print("  長期穩態分布(≈各狀態時間占比):", dict(zip(labels, np.round(stat, 3))))
    return P, labels, stat


# ------------------------------------------------------------------ #
# 4. 血糖波動圖(單一病患,含目標範圍帶)
# ------------------------------------------------------------------ #
def plot_fluctuation(patient_prefix=None, n_examples=3):
    """畫個別病患血糖波動圖。n_examples=None 表示畫全部 109 位(檔案會很多)。"""
    files = list_cgm_files()
    if patient_prefix:
        files = [f for f in files if os.path.basename(f).startswith(str(patient_prefix))]
    if n_examples is not None:
        files = files[:n_examples]
    for fp in files:
        s = load_one(fp)
        name = os.path.splitext(os.path.basename(fp))[0]
        plt.figure(figsize=(11, 3.2))
        plt.axhspan(70, 180, color="#86efac", alpha=0.35, label="Target 70–180")
        plt.axhline(70, color="#ca8a04", lw=0.8, ls="--")
        plt.axhline(180, color="#ca8a04", lw=0.8, ls="--")
        plt.plot(s["ts"], s["cgm"], color="#1d4ed8", lw=0.9)
        m = cgm_metrics(s["cgm"].values)
        plt.ylabel("CGM (mg/dL)")
        plt.title(f"CGM fluctuation — {name}  (TIR={m['TIR']:.0f}%, GMI={m['GMI']:.1f}, CV={m['CV']:.0f}%)")
        plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fluctuation_{name}.png", dpi=150); _show()
    print(f"\n[fluctuation] 已輸出 {len(files)} 張血糖波動圖 → cgm_output/fluctuation_*.png")


# ------------------------------------------------------------------ #
# 4b. 合併視圖:AGP(日內百分位)+ 示範網格 + 指標分布
# ------------------------------------------------------------------ #
def plot_agp():
    """把全部監測依『一天中的時刻』聚合,畫中位數 + IQR + 5–95% 帶(標準 AGP 圖)。"""
    rows = []
    for fp in list_cgm_files():
        s = load_one(fp)
        if len(s) < 10:
            continue
        tmin = (s["ts"].dt.hour * 60 + s["ts"].dt.minute) // 15 * 15
        rows.append(pd.DataFrame({"tmin": tmin.values, "cgm": s["cgm"].values}))
    A = pd.concat(rows, ignore_index=True)
    grid = np.arange(0, 1440, 15)
    q = A.groupby("tmin")["cgm"].quantile([.05, .25, .5, .75, .95]).unstack().reindex(grid)
    x = grid / 60.0
    plt.figure(figsize=(11, 4))
    plt.axhspan(70, 180, color="#86efac", alpha=0.30, label="Target 70–180")
    plt.fill_between(x, q[.05], q[.95], color="#93c5fd", alpha=0.35, label="5–95%")
    plt.fill_between(x, q[.25], q[.75], color="#3b82f6", alpha=0.45, label="25–75% (IQR)")
    plt.plot(x, q[.5], color="#1e3a8a", lw=2, label="Median")
    plt.axhline(70, color="#ca8a04", lw=0.8, ls="--"); plt.axhline(180, color="#ca8a04", lw=0.8, ls="--")
    plt.xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    plt.xlim(0, 24); plt.xlabel("Time of day"); plt.ylabel("CGM (mg/dL)")
    plt.title("Ambulatory Glucose Profile (AGP) — all recordings pooled")
    plt.legend(loc="upper right", fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/AGP_all.png", dpi=150); _show()
    print(f"[AGP] cgm_output/AGP_all.png")


def plot_fluctuation_grid(n=12):
    files = list_cgm_files()[:n]
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 2.3), sharey=True)
    for ax, fp in zip(axes.ravel(), files):
        s = load_one(fp); name = os.path.basename(fp).split("_")[0]
        ax.axhspan(70, 180, color="#86efac", alpha=0.25)
        ax.plot(s["ts"], s["cgm"], color="#1d4ed8", lw=0.6)
        tir = np.mean((s["cgm"] >= 70) & (s["cgm"] <= 180)) * 100
        ax.set_title(f"{name}  TIR={tir:.0f}%", fontsize=9)
        ax.tick_params(labelbottom=False, labelsize=7)
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    fig.suptitle(f"CGM fluctuation — {n} example patients", y=1.01)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fluctuation_grid{n}.png", dpi=150, bbox_inches="tight"); _show()
    print(f"[grid] cgm_output/fluctuation_grid{n}.png")


def plot_metrics_distribution(met):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (col, lab) in zip(axes, [("TIR", "TIR (%)"), ("GMI", "GMI (%)"), ("CV", "CV (%)")]):
        ax.hist(met[col], bins=20, color="#0d9488", edgecolor="white")
        ax.axvline(met[col].mean(), color="#dc2626", ls="--", lw=1.2, label=f"mean={met[col].mean():.1f}")
        ax.set_title(lab); ax.legend(fontsize=8)
    fig.suptitle("Distribution of CGM metrics across recordings", y=1.03)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/cgm_metrics_dist.png", dpi=150, bbox_inches="tight"); _show()
    met[["mean", "CV", "GMI", "TIR", "TBR", "TAR"]].describe().T.round(2).to_csv(
        f"{OUTDIR}/cgm_descriptive.csv", encoding="utf-8-sig")
    print(f"[dist] cgm_output/cgm_metrics_dist.png / cgm_descriptive.csv")


# ------------------------------------------------------------------ #
# 5. LSTM 逐步血糖預測 + 實際 vs 預測波動圖
# ------------------------------------------------------------------ #
def make_windows(series, look_back, horizon=1):
    X, y = [], []
    for i in range(len(series) - look_back - horizon + 1):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)


def lstm_forecast(patient_prefix="2000", look_back=12, horizon=1, epochs=40):
    """
    對單一病患做 next-step 血糖預測(look_back=12 ≈ 過去 3 小時 → 預測下一個 15 分鐘)。
    以時間切分:前 80% 訓練、後 20% 測試;標準化只用訓練段統計量(避免洩漏)。
    需要 tensorflow。若要對多位病患,外面包一層迴圈或改成 pooled 訓練即可。
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except Exception:
        print("\n[LSTM] 找不到 tensorflow,略過 LSTM。請在 Colab 執行:pip install tensorflow")
        return None

    files = [f for f in list_cgm_files() if os.path.basename(f).startswith(str(patient_prefix))]
    if not files:
        print(f"[LSTM] 找不到病患 {patient_prefix} 的檔案。")
        return None
    fp = files[0]
    s = load_one(fp)
    g = s["cgm"].values.astype("float32")
    n_train = int(len(g) * 0.8)

    mu, sd = g[:n_train].mean(), g[:n_train].std()
    gz = (g - mu) / sd
    Xtr, ytr = make_windows(gz[:n_train], look_back, horizon)
    Xte, yte = make_windows(gz[n_train - look_back:], look_back, horizon)  # 保留接續
    Xtr = Xtr.reshape(-1, look_back, 1)
    Xte = Xte.reshape(-1, look_back, 1)

    model = Sequential([
        LSTM(64, input_shape=(look_back, 1)),
        Dropout(0.2),
        Dense(horizon),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs, batch_size=32,
              callbacks=[es], verbose=0)

    # 測試段預測(取 horizon 的第 1 步),反標準化回 mg/dL
    pred_z = model.predict(Xte, verbose=0)[:, 0]
    pred = pred_z * sd + mu
    true = yte[:, 0] * sd + mu
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))

    # 持續性基準(predict next = current):用每個測試窗的最後一筆(即當前血糖)當預測,
    # 與 LSTM 使用完全相同的測試集,可逐窗公平對照。
    baseline = Xte[:, -1, 0] * sd + mu
    baseline_rmse = float(np.sqrt(np.mean((baseline - true) ** 2)))
    baseline_mae = float(np.mean(np.abs(baseline - true)))
    skill = (1 - rmse / baseline_rmse) * 100 if baseline_rmse > 0 else 0.0  # RMSE 相對改善(%)

    # 實際 vs 預測 波動圖(測試段)
    ts_test = s["ts"].values[n_train:n_train + len(true)]
    plt.figure(figsize=(11, 3.4))
    plt.axhspan(70, 180, color="#86efac", alpha=0.30)
    plt.plot(ts_test, true, color="#1d4ed8", lw=1.2, label="Actual")
    plt.plot(ts_test, pred, color="#dc2626", lw=1.2, ls="--", label="LSTM predicted")
    plt.plot(ts_test, baseline, color="#6b7280", lw=1.0, ls=":", label="Persistence baseline")
    plt.ylabel("CGM (mg/dL)")
    plt.title(f"LSTM next-step forecast — patient {patient_prefix}  "
              f"(LSTM RMSE={rmse:.1f} vs baseline {baseline_rmse:.1f} mg/dL)")
    plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/lstm_forecast_{patient_prefix}.png", dpi=150); _show()
    print(f"\n[LSTM] 病患 {patient_prefix}:LSTM RMSE={rmse:.1f}/MAE={mae:.1f};"
          f"持續性基準 RMSE={baseline_rmse:.1f}/MAE={baseline_mae:.1f} mg/dL"
          f"(RMSE 相對改善 {skill:.1f}%)→ cgm_output/lstm_forecast_{patient_prefix}.png")
    return dict(patient=patient_prefix, rmse=rmse, mae=mae,
                baseline_rmse=baseline_rmse, baseline_mae=baseline_mae, skill_pct=skill)


# ------------------------------------------------------------------ #
# 5b. 全體病患:LSTM vs 持續性基準 比較
# ------------------------------------------------------------------ #
def baseline_all(look_back=12):
    """
    只算「持續性基準」(predict next = current)於全部病患的測試段誤差。
    不需要 tensorflow,且完全確定(無隨機性),可先跑此函式取得基準分布。
    切分方式與 lstm_forecast 完全一致,故可直接與 LSTM 結果對照。
    """
    rows = []
    for fp in list_cgm_files():
        rid = os.path.splitext(os.path.basename(fp))[0]
        g = load_one(fp)["cgm"].values.astype("float32")
        if len(g) < look_back + 20:
            continue
        n_train = int(len(g) * 0.8)
        mu, sd = g[:n_train].mean(), g[:n_train].std()
        gz = (g - mu) / sd
        Xte, yte = make_windows(gz[n_train - look_back:], look_back, 1)
        Xte = Xte.reshape(-1, look_back, 1)
        true = yte[:, 0] * sd + mu
        base = Xte[:, -1, 0] * sd + mu
        rows.append(dict(record=rid, patient=rid.split("_")[0], n_test=len(true),
                         baseline_rmse=float(np.sqrt(np.mean((base - true) ** 2))),
                         baseline_mae=float(np.mean(np.abs(base - true)))))
    B = pd.DataFrame(rows)
    B.to_csv(f"{OUTDIR}/baseline_all.csv", index=False, encoding="utf-8-sig")
    print(f"[baseline] {len(B)} 筆監測 → cgm_output/baseline_all.csv")
    print(f"  持續性基準 RMSE:mean={B.baseline_rmse.mean():.2f}, "
          f"median={B.baseline_rmse.median():.2f}, "
          f"min={B.baseline_rmse.min():.2f}, max={B.baseline_rmse.max():.2f} mg/dL")
    return B


def lstm_forecast_all(look_back=12, horizon=1, epochs=30, n_patients=None, verbose=True):
    """
    對「全部病患」逐一訓練 LSTM 並與持續性基準比較,輸出比較表與圖。
    n_patients:限制數量(例如先設 20 試跑);None = 全部。
    注意:109 位逐一訓練約需數分鐘至數十分鐘,建議先用 n_patients 小量測試。
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except Exception:
        print("\n[LSTM-all] 找不到 tensorflow,略過。請執行:pip install tensorflow")
        return None

    files = list_cgm_files()
    if n_patients:
        files = files[:n_patients]
    rows = []
    for i, fp in enumerate(files, 1):
        rid = os.path.splitext(os.path.basename(fp))[0]
        g = load_one(fp)["cgm"].values.astype("float32")
        if len(g) < look_back + 20:
            continue
        n_train = int(len(g) * 0.8)
        mu, sd = g[:n_train].mean(), g[:n_train].std()
        gz = (g - mu) / sd
        Xtr, ytr = make_windows(gz[:n_train], look_back, horizon)
        Xte, yte = make_windows(gz[n_train - look_back:], look_back, horizon)
        Xtr = Xtr.reshape(-1, look_back, 1); Xte = Xte.reshape(-1, look_back, 1)

        model = Sequential([LSTM(64, input_shape=(look_back, 1)), Dropout(0.2), Dense(horizon)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs, batch_size=32,
                  callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)

        true = yte[:, 0] * sd + mu
        pred = model.predict(Xte, verbose=0)[:, 0] * sd + mu
        base = Xte[:, -1, 0] * sd + mu
        lr_ = float(np.sqrt(np.mean((pred - true) ** 2)))
        br_ = float(np.sqrt(np.mean((base - true) ** 2)))
        rows.append(dict(record=rid, patient=rid.split("_")[0],
                         lstm_rmse=lr_, lstm_mae=float(np.mean(np.abs(pred - true))),
                         baseline_rmse=br_, baseline_mae=float(np.mean(np.abs(base - true))),
                         skill_pct=(1 - lr_ / br_) * 100 if br_ > 0 else 0.0))
        if verbose and i % 10 == 0:
            print(f"  ...已完成 {i}/{len(files)}")

    R = pd.DataFrame(rows)
    R.to_csv(f"{OUTDIR}/lstm_vs_baseline_all.csv", index=False, encoding="utf-8-sig")
    win = (R.lstm_rmse < R.baseline_rmse).mean() * 100

    # 圖 1:散佈(基準 vs LSTM),對角線下方代表 LSTM 較佳
    lim = float(max(R.baseline_rmse.max(), R.lstm_rmse.max())) * 1.05
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(R.baseline_rmse, R.lstm_rmse, s=28, alpha=0.7, color="#dc2626", edgecolor="white")
    ax[0].plot([0, lim], [0, lim], "--", color="grey", lw=1)
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("Persistence baseline RMSE (mg/dL)")
    ax[0].set_ylabel("LSTM RMSE (mg/dL)")
    ax[0].set_title(f"LSTM vs baseline (below line = LSTM better, {win:.0f}%)")
    ax[1].hist(R.skill_pct, bins=20, color="#0d9488", edgecolor="white")
    ax[1].axvline(0, color="grey", ls="--", lw=1)
    ax[1].axvline(R.skill_pct.mean(), color="#dc2626", ls="--", lw=1.2,
                  label=f"mean={R.skill_pct.mean():.1f}%")
    ax[1].set_xlabel("RMSE improvement over baseline (%)"); ax[1].set_ylabel("count")
    ax[1].set_title("Distribution of LSTM skill"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/lstm_vs_baseline_all.png", dpi=150); _show()

    print(f"\n[LSTM-all] {len(R)} 位病患完成 → cgm_output/lstm_vs_baseline_all.csv / .png")
    print(f"  LSTM RMSE     mean={R.lstm_rmse.mean():.2f}  median={R.lstm_rmse.median():.2f}")
    print(f"  基準 RMSE     mean={R.baseline_rmse.mean():.2f}  median={R.baseline_rmse.median():.2f}")
    print(f"  平均相對改善  {R.skill_pct.mean():.1f}%  |  LSTM 勝過基準的比例:{win:.1f}%")
    return R


# ------------------------------------------------------------------ #
# 6. 主程式
# ------------------------------------------------------------------ #
def lstm_forecast_pooled(look_back=12, horizon=1, epochs=30, n_folds=5,
                         batch_size=256, units=64, verbose=True):
    """
    跨病患合併訓練的 LSTM(丙案),並以「病患分組交叉驗證」評估對全新病患的泛化能力。

    與 lstm_forecast_all()(每人各訓練一個模型)的差異:
      - 逐人版:每位病患用自己的前 80% 訓練專屬模型。訓練窗平均僅約 810 個,
               而 LSTM(64) 約有 16,900 個參數,序列較短者具有較高的過度擬合風險。
      - 本函式:把全部病患的訓練段合併成一個訓練集(約 88,000 個窗)訓練單一模型,
               參數與樣本比例回到合理範圍。

    驗證設計(關鍵):
      依「病患編號」而非「監測紀錄」分成 n_folds 組(同一病患的多次回診必在同一組)。
      每一折以其餘病患的訓練段配適模型,再預測本折病患的測試段——
      模型權重未使用測試折病患資料進行訓練；
      但每位測試病患仍使用自己的歷史訓練段估計標準化平均值與標準差。
      因此這是未見病患權重泛化搭配個人化尺度校正，不是完全零資料的 cold-start 預測。

    標準化:各病患仍以「自己訓練段」的平均與標準差正規化,預測後再還原回 mg/dL,
           故合併訓練不會因病患間血糖水準差異而失真。

    輸出:cgm_output/lstm_pooled_vs_baseline.csv、lstm_pooled_vs_perpatient.png
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except Exception:
        print("\n[LSTM-pooled] 找不到 tensorflow,略過。請執行:pip install tensorflow")
        return None

    rng = np.random.RandomState(SEED)

    # ---------- 1. 逐人準備視窗(僅用各自訓練段的統計量正規化)----------
    P = {}
    for fp in list_cgm_files():
        rid = os.path.splitext(os.path.basename(fp))[0]
        g = load_one(fp)["cgm"].values.astype("float32")
        if len(g) < look_back + 40:
            continue
        n_tr = int(len(g) * 0.8)
        mu, sd = float(g[:n_tr].mean()), float(g[:n_tr].std())
        if sd == 0:
            continue
        z = (g - mu) / sd
        Xtr, ytr = make_windows(z[:n_tr], look_back, horizon)
        Xte, yte = make_windows(z[n_tr - look_back:], look_back, horizon)
        if len(Xtr) < 30 or len(Xte) < 10:
            continue
        P[rid] = dict(pid=rid.split("_")[0], mu=mu, sd=sd, n=len(g),
                      Xtr=Xtr[..., None], ytr=ytr,
                      Xte=Xte[..., None], yte=yte)

    ids = sorted(P)
    pids = sorted({P[r]["pid"] for r in ids})
    if verbose:
        tot = sum(len(P[r]["Xtr"]) for r in ids)
        print(f"\n[LSTM-pooled] 監測 {len(ids)} 筆 / 病患 {len(pids)} 位")
        print(f"  合併訓練樣本:{tot:,} 個視窗"
              f"(逐人版平均僅 {tot // max(len(ids), 1):,} 個)")
        print(f"  驗證方式:依病患分 {n_folds} 折,每折模型完全未見該折病患")

    # ---------- 2. 依病患分折 ----------
    perm = rng.permutation(len(pids))
    fold_of_pid = {pids[j]: k % n_folds for k, j in enumerate(perm)}

    def build():
        m = Sequential([
            LSTM(units, input_shape=(look_back, 1)),
            Dropout(0.2),
            Dense(horizon),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    rows = []
    for k in range(n_folds):
        te_ids = [r for r in ids if fold_of_pid[P[r]["pid"]] == k]
        tr_ids = [r for r in ids if fold_of_pid[P[r]["pid"]] != k]
        if not te_ids or not tr_ids:
            continue
        # 訓練折內再保留約 10% 的病患作為早停用驗證集(同樣依病患分,不混同一人)
        tr_pids = sorted({P[r]["pid"] for r in tr_ids})
        n_val = max(1, len(tr_pids) // 10)
        val_pids = set(rng.permutation(tr_pids)[:n_val])
        fit_ids = [r for r in tr_ids if P[r]["pid"] not in val_pids]
        val_ids = [r for r in tr_ids if P[r]["pid"] in val_pids]

        Xf = np.concatenate([P[r]["Xtr"] for r in fit_ids])
        yf = np.concatenate([P[r]["ytr"] for r in fit_ids])
        Xv = np.concatenate([P[r]["Xtr"] for r in val_ids])
        yv = np.concatenate([P[r]["ytr"] for r in val_ids])

        model = build()
        model.fit(Xf, yf, epochs=epochs, batch_size=batch_size,
                  validation_data=(Xv, yv), verbose=0,
                  callbacks=[EarlyStopping(patience=4, restore_best_weights=True)])

        for r in te_ids:
            d = P[r]
            pred = model.predict(d["Xte"], verbose=0).ravel() * d["sd"] + d["mu"]
            act = d["yte"].ravel() * d["sd"] + d["mu"]
            base = d["Xte"][:, -1, 0] * d["sd"] + d["mu"]      # 持續性基準
            l_rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
            b_rmse = float(np.sqrt(np.mean((base - act) ** 2)))
            rows.append(dict(
                record=r, patient=d["pid"], fold=k, n_points=d["n"],
                n_train_windows=len(d["Xtr"]),
                lstm_rmse=l_rmse, lstm_mae=float(np.mean(np.abs(pred - act))),
                baseline_rmse=b_rmse, baseline_mae=float(np.mean(np.abs(base - act))),
                skill_pct=(1 - l_rmse / b_rmse) * 100 if b_rmse > 0 else np.nan))
        if verbose:
            print(f"  fold {k + 1}/{n_folds} 完成(訓練 {len(fit_ids)} 筆、"
                  f"驗證 {len(val_ids)} 筆、測試 {len(te_ids)} 筆)")

    R = pd.DataFrame(rows).sort_values("record").reset_index(drop=True)
    R.to_csv(f"{OUTDIR}/lstm_pooled_vs_baseline.csv", index=False, encoding="utf-8-sig")

    win = (R.lstm_rmse < R.baseline_rmse).mean() * 100
    print(f"\n[LSTM-pooled] {len(R)} 筆完成 → {OUTDIR}/lstm_pooled_vs_baseline.csv")
    print(f"  LSTM RMSE     mean={R.lstm_rmse.mean():.2f}  median={R.lstm_rmse.median():.2f}")
    print(f"  基準 RMSE     mean={R.baseline_rmse.mean():.2f}  median={R.baseline_rmse.median():.2f}")
    print(f"  相對改善      mean={R.skill_pct.mean():.1f}%  median={R.skill_pct.median():.1f}%")
    print(f"  勝過基準比例:{win:.1f}%  ({int((R.lstm_rmse < R.baseline_rmse).sum())}/{len(R)})")

    # 依監測長度分組:檢驗合併訓練是否解決短序列過擬合
    R["_g"] = pd.qcut(R.n_points, 4, labels=["最短", "短", "長", "最長"])
    t = R.groupby("_g", observed=True).agg(
        筆數=("record", "size"), 平均點數=("n_points", "mean"),
        改善中位數=("skill_pct", "median"),
        勝率=("skill_pct", lambda x: (x > 0).mean() * 100)).round(1)
    print("\n  依監測長度分組(檢驗短序列是否仍失效):")
    print(t.to_string())

    # ---------- 3. 與逐人版對照圖 ----------
    per_fp = f"{OUTDIR}/lstm_vs_baseline_all.csv"
    if os.path.exists(per_fp):
        A = pd.read_csv(per_fp)[["record", "skill_pct"]].rename(columns={"skill_pct": "per_patient"})
        C = R[["record", "skill_pct", "n_points"]].rename(columns={"skill_pct": "pooled"})
        M = A.merge(C, on="record")
        if len(M):
            fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
            lim = [min(M.per_patient.min(), M.pooled.min()) - 5,
                   max(M.per_patient.max(), M.pooled.max()) + 5]
            ax[0].scatter(M.per_patient, M.pooled, s=28, alpha=.7,
                          color="#0d9488", edgecolor="white")
            ax[0].plot(lim, lim, "--", color="grey", lw=1)
            ax[0].axhline(0, color="#dc2626", ls=":", lw=1)
            ax[0].axvline(0, color="#dc2626", ls=":", lw=1)
            ax[0].set_xlim(lim); ax[0].set_ylim(lim)
            ax[0].set_xlabel("Per-patient model: skill vs baseline (%)")
            ax[0].set_ylabel("Pooled model: skill vs baseline (%)")
            ax[0].set_title(f"Pooled vs per-patient "
                            f"(above line = pooled better, {(M.pooled > M.per_patient).mean()*100:.0f}%)")
            M["_g"] = pd.qcut(M.n_points, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
            gg = M.groupby("_g", observed=True)[["per_patient", "pooled"]].median()
            xx = np.arange(len(gg)); w = .36
            ax[1].bar(xx - w/2, gg.per_patient, w, color="#94a3b8", label="Per-patient")
            ax[1].bar(xx + w/2, gg.pooled, w, color="#0d9488", label="Pooled")
            ax[1].axhline(0, color="#374151", lw=1)
            ax[1].set_xticks(xx); ax[1].set_xticklabels(gg.index, fontsize=9)
            ax[1].set_ylabel("Median skill vs baseline (%)")
            ax[1].set_title("By monitoring length")
            ax[1].legend(fontsize=8.5)
            plt.tight_layout()
            plt.savefig(f"{OUTDIR}/lstm_pooled_vs_perpatient.png", dpi=150)
            _show()
            print(f"\n  對照圖 → {OUTDIR}/lstm_pooled_vs_perpatient.png")
            print(f"  合併訓練優於逐人版的比例:{(M.pooled > M.per_patient).mean()*100:.1f}%")
    return R.drop(columns=["_g"], errors="ignore")


def main():
    print("=" * 60)
    print("CGM 分析開始")
    print("=" * 60)

    met = build_metrics_table()          # 1. 指標(可併回 Summary)
    markov_matrix(n_states=3)             # 2a. 3 態馬可夫
    markov_matrix(n_states=5)             # 2b. 5 態馬可夫
    plot_agp()                            # 3a. AGP:全部病患合併(日內百分位)
    plot_fluctuation_grid(n=12)           # 3b. 12 位示範網格
    plot_metrics_distribution(met)        # 3c. TIR/GMI/CV 分布 + 描述統計
    plot_fluctuation(n_examples=3)        # 3d. 個別波動圖(改 None 可畫全部 109 位)
    lstm_forecast(patient_prefix="2000")  # 4. LSTM 預測 + 實際vs預測圖(單一病患)
    baseline_all()                        # 5a. 全體病患的持續性基準(不需 tensorflow)
    # 5b. 全體病患 LSTM vs 基準(較耗時;可先用 n_patients=20 試跑)
    lstm_forecast_all(epochs=30, n_patients=None)
    # 5c. 跨病患合併訓練 + 病患分組交叉驗證(丙案);需先跑完 5b 才能產生對照圖
    lstm_forecast_pooled(epochs=30, n_folds=5)

    print(f"\n完成。所有輸出在 ./{OUTDIR}/")
    print("圖:markov_3state.png、markov_5state.png、fluctuation_*.png、lstm_forecast_*.png")
    print("表:cgm_metrics.csv、markov_3state.csv、markov_5state.csv")


if __name__ == "__main__":
    main()
