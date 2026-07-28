# -*- coding: utf-8 -*-
"""
糖尿病餘生醫療成本 —— 蒙地卡羅模擬(機率敏感度分析 PSA)
================================================================
在確定性版(lifetime_cost.py)之上,對每位病患模擬 N 次「餘生醫療成本」(自當前年齡至 80 歲),
把三種不確定性一起隨機抽樣,輸出成本的「平均 + 95% 區間 + 分布」:

  (1) 併發症狀態不確定性:依分類模型的 P(大血管)、P(小血管) 抽 Bernoulli,
      每次落入 無 / 僅大 / 僅小 / 大+小 四組之一(來自你的模型,非人工假設)。
  (2) 成本參數不確定性:各組年成本以 Gamma 分布抽樣(均值=陳興寶 2003 表 1,
      變異係數 COST_CV 為 PSA 假設,可調;文獻未提供 SE 時的標準作法)。
  (3) 物價校正不確定性:2001→約2023 的醫療 CPI 校正倍數以 Gamma 抽樣(均值 INFLATION_MEAN、
      變異係數 INFLATION_CV;預設 1.8/0.15,依中國醫療保健類 CPI 估計,請以統計年鑑精確值取代)。

成本點估計來源(已核對原文表 1):
  陳興寶等(2003)《中国糖尿病杂志》11(4):238-241,上海復旦大學公共衛生學院。

時界到 HORIZON_AGE 歲(自「當前年齡」起算)、折現率 DISCOUNT_RATE(健康經濟常用 3%)。

★ 為何只估「餘生前瞻成本」(現在→80),而不估「全病程終身成本」(發病→80)★
  第 2 型糖尿病為成年後罹患(非天生),理論上完整終身成本應自「發病年齡 = 當前年齡 − 罹病時間」
  起算。但要估「發病到現在」這段歷史成本,必須知道病患過去各年度的併發症狀態;本資料為橫斷面、
  無逐年追蹤,若以「當前狀態套用到過去每一年」會高估早期(剛發病時多半尚無併發症),若以「線性
  爬升」補之則等於用未經實證的假設冒充資料。為守住「不以假設冒充資料」的原則,本模型只估算每個
  數字皆有依據(年齡、模型風險、成本、折現)的『餘生前瞻成本』。
  ★ 罹病時間(Duration)並未浪費:它是分類模型中小血管併發症最強的預測因子(OR≈3.25),
    亦即「罹病越久 → 併發症機率越高 → 餘生成本越高」——Duration 透過影響風險預測來影響成本,
    而非拿去乘一段虛構的歷史成本。此為其正確且完全以資料驅動的用法。
  本模型另以「當前預期狀態」外推未來,未建模逐年病程進展(那需 CORE/UKPDS 級微模擬),屬一階近似。

依賴:pandas, numpy, scikit-learn, matplotlib;需與 diabetes_deterioration_pipeline.py 同目錄。

----------------------------------------------------------------
【重要：本成本程式目前僅可視為探索性情境分析】
----------------------------------------------------------------
1. 幣值基期:
   文獻發表於 2003 年，但表中成本為 2001 年人民幣，因此 CPI 轉換基期應為 2001，
   不是 2003。

2. 物價換算:
   INFLATION_MEAN=1.8 目前只是 placeholder，不可當成已驗證參數。
   正式版本應使用上海市官方「醫療保健類 CPI」逐年鏈結 2001→2025：
       factor = Π(CPI_t / 100)
       cost_2025 = cost_2001 × factor
   並在報告列出每年指數、官方表名、網址/年鑑、下載日期及計算表。
   若不同年份的 CPI 分類口徑曾改變，需另列可比性限制。

3. 不可重複計算通膨:
   若成本已換算為 2025 年不變人民幣，未來年度應使用實質折現率，
   不應再額外套一次一般通膨成長。

4. 風險來源限制:
   p_macro、p_micro 來自橫斷面「目前是否已有併發症」分類，
   不是逐年新發併發症機率。把同一機率一路外推到 80 歲的假設很強，
   因此目前不宜稱為已驗證的個人餘生成本預測。

5. 狀態獨立假設:
   分別抽樣 Macro ~ Bernoulli(p_macro) 與 Micro ~ Bernoulli(p_micro)，
   隱含給定 X 後兩種併發症條件獨立；此假設尚未由資料驗證。

6. 重複病患:
   目前迴圈依臨床「紀錄」運算。若 109 筆來自約 100 位病患，
   母體總成本可能重複計算同一人。病患層級與母體總額正式輸出前，
   應先建立每位病患一列的資料。

較穩健的近期成果:
   優先改為「2025 年度成本情境分析」；若保留未來成本，可先做固定 5 年情境，
   並明確標示狀態維持不變。完整餘生成本需年度病程轉移率與死亡率。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython(); _NB = True
except NameError:
    matplotlib.use("Agg"); _NB = False
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

_av = {f.name for f in _fm.fontManager.ttflist}
for _f in ["Microsoft JhengHei", "Microsoft YaHei", "PingFang TC", "Heiti TC",
           "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]:
    if _f in _av:
        plt.rcParams["font.sans-serif"] = [_f]; break
plt.rcParams["axes.unicode_minus"] = False

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from diabetes_deterioration_pipeline import load_data

SEED = 42
rng = np.random.default_rng(SEED)
OUTDIR = "output_cost_mc"
os.makedirs(OUTDIR, exist_ok=True)

# ---- 設定(可調)----
N_SIM = 5000               # 每位病患的模擬次數
COST_TYPE = "direct"       # "direct"=年直接醫療費用;"total"=年總費用
COST_CV = 0.30             # 成本 PSA 變異係數(假設;可調)
# 物價校正:陳興寶成本為 2001 年幣值。中國「醫療保健類 CPI」漲幅溫和(年增多為
# +0.4~3%,個別年份 ~+7%),累計 2001→約2023 約 1.5–2×。此處預設 1.8×(可調),
# 請以《中國統計年鑑》醫療保健類 CPI 之精確累計值取代。切勿用「人均衛生支出成長」
# (~13×),因其含使用量/技術成長,會高估「同一項併發症」之價格。
INFLATION_MEAN = 1.8       # 2001→約2023 醫療 CPI 校正倍數(醫療 CPI 基準;請自行更新)
INFLATION_CV = 0.15        # 校正倍數不確定性(涵蓋約 1.4–2.3×)
DISCOUNT_RATE = 0.03
HORIZON_AGE = 80           # 餘生時界:估算至 80 歲(成本自「當前年齡」起算,見下)
RISK_TABLE = "output_risk/patient_risk_table.csv"   # 有則併入象限
# 註:第 2 型糖尿病為成年後才罹患(非天生),故成本自病患「當前年齡」向前累計至 80 歲,
#     屬「餘生前瞻醫療成本」,而非自出生起的終身總額。

# ---- 陳興寶(2003)表 1 四組年成本(2001 RMB,已核對)----
COST = {
    "direct": {"none": 3726.36, "macro": 15373.87, "micro": 11842.05, "both": 38580.24},
    "total":  {"none": 4773.83, "macro": 18836.98, "micro": 15101.63, "both": 44465.52},
}


def oof_prob(X, y, groups):
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                              solver="liblinear", random_state=SEED))])
    return cross_val_predict(lr, X, y, cv=GroupKFold(5), groups=groups,
                             method="predict_proba")[:, 1]


def gamma_samples(mean, cv, size):
    """以指定均值與變異係數抽 Gamma(保證正值)。cv=0 → 回傳常數。"""
    if cv <= 0:
        return np.full(size, mean, dtype=float)
    k = 1.0 / (cv ** 2)
    theta = mean * (cv ** 2)
    return rng.gamma(k, theta, size=size)


def main():
    d, med, targets, pid = load_data()
    raw = pd.read_excel("Shanghai_T2DM_Summary.xlsx", sheet_name="T2DM")
    record_id = raw["Patient Number"].astype(str).values
    age = d["Age"].fillna(d["Age"].median()).values

    p_macro = oof_prob(d, targets["Macrovascular"].astype(int), pid)
    p_micro = oof_prob(d, targets["Microvascular"].astype(int), pid)
    c = COST[COST_TYPE]
    r = DISCOUNT_RATE

    # FIXME：以下目前逐「紀錄」迴圈；若同一病患有多次回診，母體總成本會重複計人。
    # 正式病患層級/母體結果前，應先依 patient_id 選定一筆基準或最新紀錄。
    rows = []
    all_pop = np.zeros(N_SIM)                 # 母體每次模擬的總成本(供母體分布)
    for i in range(len(d)):
        years = max(HORIZON_AGE - age[i], 1)
        annuity = (1 - (1 + r) ** (-years)) / r if r > 0 else years

        # (1) 併發症狀態:依模型機率抽 Bernoulli
        # 注意：這隱含 Macro 與 Micro 在給定 X 後條件獨立，尚未被本資料驗證。
        mac = rng.random(N_SIM) < p_macro[i]
        mic = rng.random(N_SIM) < p_micro[i]
        # 每個狀態抽該組成本(Gamma),再依抽到的狀態選取
        cost_state = np.select(
            [mac & mic, mac & ~mic, ~mac & mic],
            [gamma_samples(c["both"], COST_CV, N_SIM),
             gamma_samples(c["macro"], COST_CV, N_SIM),
             gamma_samples(c["micro"], COST_CV, N_SIM)],
            default=gamma_samples(c["none"], COST_CV, N_SIM))
        # (3) 通膨倍數
        infl = gamma_samples(INFLATION_MEAN, INFLATION_CV, N_SIM)

        lifetime = cost_state * infl * annuity            # N_SIM 條終身成本
        all_pop += lifetime
        rows.append(dict(
            record=record_id[i], age=int(round(age[i])),
            P_macro=round(float(p_macro[i]), 3), P_micro=round(float(p_micro[i]), 3),
            lifetime_mean=float(lifetime.mean()),
            lifetime_lo=float(np.percentile(lifetime, 2.5)),
            lifetime_hi=float(np.percentile(lifetime, 97.5)),
            annual_mean=float((cost_state * infl).mean()),
        ))

    out = pd.DataFrame(rows)
    if os.path.exists(RISK_TABLE):
        rt = pd.read_csv(RISK_TABLE)[["record", "quadrant", "long_group"]]
        out = out.merge(rt, on="record", how="left")
    out_round = out.copy()
    for col in ["lifetime_mean", "lifetime_lo", "lifetime_hi", "annual_mean"]:
        out_round[col] = out_round[col].round(0)
    out_round.to_csv(f"{OUTDIR}/patient_lifetime_cost_mc.csv", index=False, encoding="utf-8-sig")

    unit = "年直接醫療費用" if COST_TYPE == "direct" else "年總費用"
    print(f"蒙地卡羅:N={N_SIM}/人;成本源 陳興寶 2003 表 1({unit},2001 RMB)")
    print(f"成本CV={COST_CV}、通膨={INFLATION_MEAN}±(CV {INFLATION_CV})、折現 {r:.0%}、到 {HORIZON_AGE} 歲\n")
    print("每人餘生成本平均(至80歲,RMB):  mean=%.0f  median=%.0f" %
          (out.lifetime_mean.mean(), out.lifetime_mean.median()))
    print("母體餘生總成本(至80歲,RMB):    mean=%.0f  95%%CI=[%.0f, %.0f]" %
          (all_pop.mean(), np.percentile(all_pop, 2.5), np.percentile(all_pop, 97.5)))

    # ---- 圖 1:各風險象限的終身成本(平均 + 95% 區間誤差棒)----
    if "quadrant" in out.columns:
        g = out.groupby("quadrant").agg(
            n=("record", "size"),
            mean=("lifetime_mean", "mean"),
            lo=("lifetime_lo", "mean"),
            hi=("lifetime_hi", "mean")).reindex(
            ["A 立即介入(長高+短高)", "B 慢性追蹤(長高+短低)",
             "C 血糖波動注意(長低+短高)", "D 常規追蹤(長低+短低)"]).dropna()
        print("\n各風險象限餘生成本(平均 [95% 區間平均],萬元):")
        for q, row in g.iterrows():
            print(f"  {q}: {row['mean']/1e4:.1f} 萬 [{row['lo']/1e4:.1f}, {row['hi']/1e4:.1f}] (n={int(row['n'])})")
        plt.figure(figsize=(8.2, 4.6))
        x = np.arange(len(g))
        colors = ["#dc2626", "#f59e0b", "#3b82f6", "#16a34a"][:len(g)]
        yerr = np.vstack([(g["mean"] - g["lo"]) / 1e4, (g["hi"] - g["mean"]) / 1e4])
        plt.bar(x, g["mean"] / 1e4, color=colors, alpha=0.85)
        plt.errorbar(x, g["mean"] / 1e4, yerr=yerr, fmt="none", ecolor="#333", capsize=5, lw=1.2)
        for xi, m in zip(x, g["mean"]):
            plt.text(xi, m / 1e4, f"{m/1e4:.1f}萬", ha="center", va="bottom", fontsize=10)
        plt.xticks(x, g.index, rotation=12, fontsize=9)
        plt.ylabel("預估餘生醫療成本(萬元 RMB)")
        plt.title(f"各風險象限餘生成本(至80歲):蒙地卡羅平均 + 95% 區間(成本源:陳興寶 2003)")
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_cost_by_quadrant.png", dpi=150)
        if _NB: plt.show()
        plt.close()

    # ---- 圖 2:每位病患終身成本(依平均排序,含 95% 區間)----
    s = out.sort_values("lifetime_mean").reset_index(drop=True)
    plt.figure(figsize=(9, 4.4))
    xx = np.arange(len(s))
    plt.fill_between(xx, s.lifetime_lo / 1e4, s.lifetime_hi / 1e4,
                     color="#c4b5fd", alpha=0.6, label="95% 區間")
    plt.plot(xx, s.lifetime_mean / 1e4, color="#6d28d9", lw=1.5, label="平均")
    plt.xlabel("病患(依餘生成本平均排序)"); plt.ylabel("餘生醫療成本(萬元 RMB)")
    plt.title(f"每位病患餘生成本(至80歲):蒙地卡羅平均與 95% 區間(N={N_SIM}/人)")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/mc_cost_per_patient.png", dpi=150)
    if _NB: plt.show()
    plt.close()

    # ---- 圖 3:母體總終身成本的模擬分布 ----
    plt.figure(figsize=(7, 4))
    plt.hist(all_pop / 1e6, bins=30, color="#0d9488", edgecolor="white")
    for pct, lab, col in [(2.5, "2.5%", "#dc2626"), (50, "中位數", "#111"), (97.5, "97.5%", "#dc2626")]:
        v = np.percentile(all_pop, pct)
        plt.axvline(v / 1e6, color=col, ls="--", lw=1.1)
    plt.xlabel("母體餘生總醫療成本(百萬元 RMB)"); plt.ylabel("模擬次數")
    plt.title(f"母體餘生總成本(至80歲)的蒙地卡羅分布(N={N_SIM} 次)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_population_total.png", dpi=150)
    if _NB: plt.show()
    plt.close()

    # ---- 圖 4:成本集中度勞倫茲曲線 + Gini 係數 ----
    cc = np.sort(out["lifetime_mean"].values)
    n = len(cc)
    cum_p = np.arange(1, n + 1) / n
    cum_c = np.cumsum(cc) / cc.sum()
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.x/1.x 皆可
    gini = 1 - 2 * _trapz(cum_c, cum_p)
    top10 = cc[::-1][:max(1, round(n * 0.10))].sum() / cc.sum() * 100
    top20 = cc[::-1][:max(1, round(n * 0.20))].sum() / cc.sum() * 100
    plt.figure(figsize=(6.2, 5.6))
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, label="完全平均線")
    plt.plot(np.r_[0, cum_p], np.r_[0, cum_c], color="#6d28d9", lw=2.2,
             label=f"實際分布(Gini={gini:.2f})")
    plt.fill_between(np.r_[0, cum_p], np.r_[0, cum_c], np.r_[0, cum_p],
                     color="#c4b5fd", alpha=0.45)
    for k, share, col in [(10, top10, "#dc2626"), (20, top20, "#ea580c")]:
        x = 1 - k / 100
        y = np.interp(x, cum_p, cum_c)
        plt.plot([x, x], [0, y], ":", color=col, lw=1.2)
        plt.plot([x, 1], [y, y], ":", color=col, lw=1.2)
        plt.annotate(f"最貴前{k}%\n占 {share:.0f}%", xy=(x, y), xytext=(x - 0.34, y + 0.05),
                     fontsize=9, color=col, arrowprops=dict(arrowstyle="->", color=col, lw=1))
    plt.xlabel("累積病患比例(由成本低到高)"); plt.ylabel("累積餘生成本比例")
    plt.title("餘生醫療成本集中度:勞倫茲曲線")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_lorenz.png", dpi=150)
    if _NB: plt.show()
    plt.close()
    print(f"[集中度] Gini={gini:.3f};最貴前 10% 占 {top10:.1f}%、前 20% 占 {top20:.1f}% → mc_lorenz.png")

    print(f"\n完成。輸出於 ./{OUTDIR}/  (patient_lifetime_cost_mc.csv、mc_*.png)")
    print("提醒:已用 INFLATION_MEAN=1.8(醫療 CPI 估計)校正至約 2023 年幣值;請以統計年鑑醫療保健類 CPI 精確累計值取代。")


if __name__ == "__main__":
    main()
