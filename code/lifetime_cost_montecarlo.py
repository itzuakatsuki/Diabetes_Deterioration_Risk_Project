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

from diabetes_deterioration_pipelin import load_data

SEED = 42
rng = np.random.default_rng(SEED)
OUTDIR = "output_cost_mc"
os.makedirs(OUTDIR, exist_ok=True)

# ---- 設定(可調)----
N_SIM = 5000               # 每位病患的模擬次數
COST_TYPE = "direct"       # "direct"=年直接醫療費用;"total"=年總費用
COST_CV = 0.30             # 成本 PSA 變異係數(假設;可調)
# ---- 物價校正:以《上海統計年鑑》醫療保健類 CPI 逐年連乘(2001 → 2024 年幣值)----
# 陳興寶成本為 2001 年幣值。採上海口徑而非全國,係因成本來源(復旦大學)與 CGM 樣本
# 同為上海,口徑一致性較佳。切勿用「人均衛生支出成長」(~13×),因其含使用量/技術
# 成長,會高估「同一項併發症」之價格。
#
# 環比指數(上年=100),來源:《上海統計年鑑》價格章「居民消費價格(分類)指數」
#   2002-2015 取自表 9.2(1991~2015)之「醫療保健和個人用品」大類
#   2016-2018 取自表 9.2(2016~2018)、2019-2020 表 8.2(2018~2020)
#   2021-2022 表 8.2(2021~2022)、2023-2024 表 8.2(2022~2024)
# ★ 口徑斷點:2015 年(含)以前為「醫療保健和個人用品」,2016 年起「個人用品」移出、
#   成為獨立之「醫療保健」大類。此為現有公布資料範圍內的最佳做法,報告需載明。
# ★ 重疊年交叉核對:2018 兩表皆 102.4、2022 兩表皆 102.1,佐證分段串接無誤。
MEDICAL_CPI = {
    2002: 97.6,  2003: 100.0, 2004: 100.0, 2005: 100.3, 2006: 101.1, 2007: 100.2,
    2008: 103.1, 2009: 99.4,  2010: 103.7, 2011: 104.1, 2012: 100.6, 2013: 100.0,
    2014: 100.4, 2015: 99.3,  2016: 109.0, 2017: 106.6, 2018: 102.4, 2019: 103.3,
    2020: 101.2, 2021: 98.9,  2022: 102.1, 2023: 100.2, 2024: 99.2,
}
COST_BASE_YEAR = 2001      # 成本資料的幣值年(非論文出版年 2003)
TARGET_YEAR = 2024         # 校正到哪一年的幣值


def cumulative_inflation(base_year=COST_BASE_YEAR, target_year=TARGET_YEAR):
    """
    累計倍數 = Π_{y=base+1}^{target} CPI[y]/100。
    環比指數 CPI[y] 表示「y 年相對 y-1 年」,故 CPI[2001] 描述的是 2000→2001,
    成本既為 2001 年幣值即不應計入;2001→2024 跨 23 年,對應 2002 至 2024 共 23 項。
    (連乘而非各年漲幅相加,為國家統計局確認之算法。)
    """
    years = list(range(base_year + 1, target_year + 1))
    missing = [y for y in years if y not in MEDICAL_CPI]
    if missing:
        raise ValueError(f"MEDICAL_CPI 缺少年度:{missing}")
    mult = 1.0
    for y in years:
        mult *= MEDICAL_CPI[y] / 100.0
    return mult


INFLATION_MEAN = cumulative_inflation()   # 2001→2024 = 1.3737(年均 +1.39%)
# 逐年指數為統計年鑑公布之確定數值,非待估參數,故不再對其加設分布。
# 口徑選擇(上海 vs 全國)之影響改以敏感度分析呈現。
INFLATION_CV = 0.0
DISCOUNT_RATE = 0.03
HORIZON_AGE = 80           # 僅在 HORIZON_MODE="fixed_age" 時使用

# [修正 1] 原程式 years = max(HORIZON_AGE - age, 1),但資料中有 8 筆年齡 >= 80
#          (83,84,84,85,85,85,92,97),這些人被強制算成「餘生 1 年」,邏輯不成立且
#          嚴重低估(83 歲者年金因子 0.971 對生命表餘命 7 年的 6.230,差 6.4 倍)。
#          改用生命表平均餘命:對年輕病患結果與原作法相近(57 歲原 23 年 vs 餘命約 24 年),
#          同時正確處理高齡者。保留 "fixed_age" 模式可還原原作法作對照。
HORIZON_MODE = "life_table"    # "life_table"(建議) 或 "fixed_age"(原作法)
# 平均餘命(年)—— 量級估計,建議以中國/上海官方生命表精確值取代
LIFE_EXPECTANCY = {20: 58, 30: 49, 40: 39, 50: 30, 55: 26, 60: 21.5,
                   65: 17.5, 70: 14, 75: 10.8, 80: 8, 85: 5.8, 90: 4.1, 95: 3.0, 100: 2.0}
RISK_TABLE = "output_risk/patient_risk_table.csv"   # 有則併入象限

# ---- 併發症機率的類別權重設定(基準情境 vs 敏感度情境)----
# 分類模型以 class_weight="balanced" 配適,可提升少數類的召回、對排序(AUC)有利;
# 但該設定會提高少數類權重、上移截距,使預測機率系統性高於實際盛行率。
# 本成本模型以 rng.random() < p 抽 Bernoulli,吃的是「機率絕對值」而非排序,
# 故此偏誤會直接傳遞至成本。實測(本資料):
#   大血管 實際盛行率 36.7% / balanced 預測均值 43.4%(+6.7 pp)
#   小血管 實際盛行率 24.8% / balanced 預測均值 34.1%(+9.4 pp)
# 因此本研究以 balanced 為基準情境(與報告 4.3、4.9 之分類模型一致),
# 另以未加權模型作為敏感度情境,呈現校準差異對成本估計的影響區間。
CLASS_WEIGHT = "balanced"          # 基準情境;敏感度情境為 None
CLASS_WEIGHT_SCENARIOS = ["balanced", None]
# 註:第 2 型糖尿病為成年後才罹患(非天生),故成本自病患「當前年齡」向前累計,
#     屬「餘生前瞻醫療成本」,而非自出生起的終身總額。

# ---- 陳興寶(2003)表 1 四組年成本(2001 RMB,已核對)----
COST = {
    "direct": {"none": 3726.36, "macro": 15373.87, "micro": 11842.05, "both": 38580.24},
    "total":  {"none": 4773.83, "macro": 18836.98, "micro": 15101.63, "both": 44465.52},
}


def oof_prob(X, y, groups, class_weight=None):
    """out-of-fold 陽性機率。class_weight 未指定時採 CLASS_WEIGHT(基準情境)。"""
    cw = CLASS_WEIGHT if class_weight == "__default__" else class_weight
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000, class_weight=cw,
                                              solver="liblinear", random_state=SEED))])
    return cross_val_predict(lr, X, y, cv=GroupKFold(5), groups=groups,
                             method="predict_proba")[:, 1]


def simulate_costs(p_macro, p_micro, age, is_first, seed=SEED):
    """給定併發症機率,執行蒙地卡羅並回傳(每人成本平均陣列, 母體總成本 N_SIM 條)。"""
    rg = np.random.default_rng(seed)
    c = COST[COST_TYPE]; r = DISCOUNT_RATE

    def gam(mean, cv, size):
        if cv <= 0:
            return np.full(size, mean, dtype=float)
        return rg.gamma(1.0 / cv ** 2, mean * cv ** 2, size=size)

    cost_draw = {k: gam(v, COST_CV, N_SIM) for k, v in c.items()}
    infl_draw = gam(INFLATION_MEAN, INFLATION_CV, N_SIM)
    pop = np.zeros(N_SIM); per = []
    for i in range(len(p_macro)):
        ann = (1 - (1 + r) ** (-remaining_years(age[i]))) / r
        mac = rg.random(N_SIM) < p_macro[i]
        mic = rg.random(N_SIM) < p_micro[i]
        cs = np.select([mac & mic, mac & ~mic, ~mac & mic],
                       [cost_draw["both"], cost_draw["macro"], cost_draw["micro"]],
                       default=cost_draw["none"])
        lt = cs * infl_draw * ann
        per.append(lt.mean())
        if is_first[i]:
            pop += lt
    return np.array(per), pop


def sensitivity_class_weight(d, targets, pid, age, is_first):
    """
    敏感度分析:比較 class_weight="balanced"(基準)與 None(未加權)兩種機率設定
    對餘生成本估計的影響,並同時報告兩者的校準品質。
    輸出 cost_sensitivity_class_weight.csv。
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y_mac = targets["Macrovascular"].astype(int).values
    y_mic = targets["Microvascular"].astype(int).values
    rows = []
    for cw in CLASS_WEIGHT_SCENARIOS:
        pm = oof_prob(d, y_mac, pid, class_weight=cw)
        pi = oof_prob(d, y_mic, pid, class_weight=cw)
        per, pop = simulate_costs(pm, pi, age, is_first)
        rows.append(dict(
            class_weight=str(cw),
            macro_prev=y_mac.mean() * 100, macro_pred=pm.mean() * 100,
            micro_prev=y_mic.mean() * 100, micro_pred=pi.mean() * 100,
            macro_auc=roc_auc_score(y_mac, pm), micro_auc=roc_auc_score(y_mic, pi),
            macro_brier=brier_score_loss(y_mac, pm), micro_brier=brier_score_loss(y_mic, pi),
            cost_mean=per.mean(), cost_median=float(np.median(per)),
            pop_total=pop.mean(),
            pop_lo=float(np.percentile(pop, 2.5)), pop_hi=float(np.percentile(pop, 97.5))))
    S = pd.DataFrame(rows)
    S.round(4).to_csv(f"{OUTDIR}/cost_sensitivity_class_weight.csv",
                      index=False, encoding="utf-8-sig")

    print("\n=== 敏感度分析:併發症機率之類別權重設定 ===")
    print(f"{'設定':<16}{'大血管預測':>11}{'小血管預測':>11}{'AUC(大/小)':>14}"
          f"{'每人平均':>11}{'母體總計':>11}")
    print("-" * 76)
    print(f"{'實際盛行率':<14}{S.macro_prev[0]:>10.1f}%{S.micro_prev[0]:>10.1f}%"
          f"{'—':>14}{'—':>11}{'—':>11}")
    for _, x in S.iterrows():
        lab = "balanced(基準)" if x.class_weight == "balanced" else "未加權(敏感度)"
        print(f"{lab:<14}{x.macro_pred:>10.1f}%{x.micro_pred:>10.1f}%"
              f"{x.macro_auc:>7.3f}/{x.micro_auc:.3f}{x.cost_mean:>11,.0f}"
              f"{x.pop_total/1e4:>9,.0f}萬")
    b, u = S.iloc[0], S.iloc[1]
    print(f"\n  兩情境差距:每人平均 {(u.cost_mean/b.cost_mean-1)*100:+.1f}%、"
          f"母體總計 {(u.pop_total/b.pop_total-1)*100:+.1f}%")
    print(f"  校準品質(Brier,越低越好):balanced {b.macro_brier:.4f}/{b.micro_brier:.4f}、"
          f"未加權 {u.macro_brier:.4f}/{u.micro_brier:.4f}")
    print("  ★ 未加權模型之預測均值較接近實際盛行率、Brier 較低,故基準情境之成本估計")
    print("    可視為上界;兩情境的象限相對倍數與集中度指標不受影響。")
    return S


def gamma_samples(mean, cv, size):
    """以指定均值與變異係數抽 Gamma(保證正值)。cv=0 → 回傳常數。"""
    if cv <= 0:
        return np.full(size, mean, dtype=float)
    k = 1.0 / (cv ** 2)
    theta = mean * (cv ** 2)
    return rng.gamma(k, theta, size=size)


def horizon_label():
    """[修正 1 配套] 圖表與訊息一律用這個字串,避免標題寫死「至 80 歲」與實際時界不符。"""
    return "生命表平均餘命" if HORIZON_MODE == "life_table" else f"至 {HORIZON_AGE} 歲"


def money_label():
    """金額幣值年說明。"""
    return f"{TARGET_YEAR} 年人民幣"


def remaining_years(age):
    """[修正 1] 依 HORIZON_MODE 回傳成本累計年數。"""
    if HORIZON_MODE == "fixed_age":
        return max(HORIZON_AGE - age, 1)          # 原作法(高齡者會被壓成 1 年)
    ages = np.array(sorted(LIFE_EXPECTANCY))
    vals = np.array([LIFE_EXPECTANCY[a] for a in ages])
    return float(np.interp(age, ages, vals))


def main():
    d, med, targets, pid, _ = load_data()   # [修正] load_data 現多回傳 d_raw
    raw = pd.read_excel("Shanghai_T2DM_Summary.xlsx", sheet_name="T2DM")
    record_id = raw["Patient Number"].astype(str).values
    age = d["Age"].fillna(d["Age"].median()).values

    p_macro = oof_prob(d, targets["Macrovascular"].astype(int), pid, class_weight=CLASS_WEIGHT)
    p_micro = oof_prob(d, targets["Microvascular"].astype(int), pid, class_weight=CLASS_WEIGHT)
    c = COST[COST_TYPE]
    r = DISCOUNT_RATE

    # [修正 3] 年成本與物價倍數屬「母體參數不確定性」——其真值對全體病患只有一個,
    #          不會因人而異。原程式在病患迴圈內各自抽樣,109 次獨立抽樣觸發大數法則、
    #          誤差互相抵消,使母體區間相對寬度由 59.5% 縮成 5.6%(實測),
    #          母體 95% 區間因而嚴重低估。改為迴圈外各抽 N_SIM 條、每次模擬全體共用。
    #          併發症狀態(Bernoulli)維持在迴圈內,那才是真正的個體層級不確定性。
    cost_draw = {k: gamma_samples(v, COST_CV, N_SIM) for k, v in c.items()}
    infl_draw = gamma_samples(INFLATION_MEAN, INFLATION_CV, N_SIM)

    # [修正 2] 母體加總只計入每位病患的首筆紀錄。原程式逐 109 筆累加,
    #          但資料為 109 筆紀錄 / 100 位病患(8 位共 9 筆重複回診),
    #          同一病患的多次回診被重複計入母體總成本。個人估計仍全數輸出。
    is_first = ~pd.Series(pid.values).duplicated(keep="first").values

    rows = []
    all_pop = np.zeros(N_SIM)                 # 母體每次模擬的總成本(供母體分布)
    for i in range(len(d)):
        years = remaining_years(age[i])                    # [修正 1]
        annuity = (1 - (1 + r) ** (-years)) / r if r > 0 else years

        # (1) 併發症狀態:依模型機率抽 Bernoulli
        mac = rng.random(N_SIM) < p_macro[i]
        mic = rng.random(N_SIM) < p_micro[i]
        # 每個狀態抽該組成本(Gamma),再依抽到的狀態選取
        cost_state = np.select(
            [mac & mic, mac & ~mic, ~mac & mic],
            [cost_draw["both"], cost_draw["macro"], cost_draw["micro"]],
            default=cost_draw["none"])                    # [修正 3] 全體共用抽樣

        lifetime = cost_state * infl_draw * annuity       # N_SIM 條餘生成本
        if is_first[i]:                                   # [修正 2] 只計首筆
            all_pop += lifetime
        rows.append(dict(
            record=record_id[i], age=int(round(age[i])),
            years_horizon=round(years, 1), counted_in_population=bool(is_first[i]),
            P_macro=round(float(p_macro[i]), 3), P_micro=round(float(p_micro[i]), 3),
            lifetime_mean=float(lifetime.mean()),
            lifetime_lo=float(np.percentile(lifetime, 2.5)),
            lifetime_hi=float(np.percentile(lifetime, 97.5)),
            annual_mean=float((cost_state * infl_draw).mean()),
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
    _n = TARGET_YEAR - COST_BASE_YEAR
    print(f"物價校正={INFLATION_MEAN:.4f}(上海醫療保健類 CPI {COST_BASE_YEAR}→{TARGET_YEAR} "
          f"逐年連乘 {_n} 項,年均 {INFLATION_MEAN ** (1/_n) - 1:+.2%})")
    print(f"成本CV={COST_CV}、折現 {r:.0%}、時界={'生命表餘命' if HORIZON_MODE=='life_table' else f'至 {HORIZON_AGE} 歲'}\n")
    print(f"每人餘生成本平均({horizon_label()},{money_label()}):  mean=%.0f  median=%.0f" %
          (out.lifetime_mean.mean(), out.lifetime_mean.median()))
    print(f"母體餘生總成本({horizon_label()},{money_label()}):    mean=%.0f  95%%模擬區間=[%.0f, %.0f]" %
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
        # 誤差線右移 0.22、長條收窄為 0.62,避免誤差線貫穿數值標籤
        plt.bar(x, g["mean"] / 1e4, color=colors, alpha=0.85, width=0.62)
        plt.errorbar(x + 0.22, g["mean"] / 1e4, yerr=yerr, fmt="none",
                     ecolor="#333", capsize=5, lw=1.2)
        for xi, m in zip(x, g["mean"]):
            plt.text(xi - 0.06, m / 1e4 + 1.5, f"{m/1e4:.1f} 萬", ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
        plt.margins(y=0.16)
        plt.xticks(x, g.index, rotation=12, fontsize=9)
        plt.ylabel(f"預估餘生醫療成本(萬元,{money_label()})")
        plt.title(f"各風險象限餘生成本({money_label()};時界:{horizon_label()})\n"
                  f"蒙地卡羅平均 + 95% 模擬區間(成本源:陳興寶 2003)")
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_cost_by_quadrant.png", dpi=150)
        if _NB: plt.show()
        plt.close()

    # ---- 圖 2:每位病患終身成本(依平均排序,含 95% 區間)----
    # [修正 2 配套] 此圖與勞倫茲曲線、母體總計一致,皆以去重後的病患為單位;
    #               全部 109 筆的個人估計仍完整輸出於 CSV。
    s = out[out["counted_in_population"]].sort_values("lifetime_mean").reset_index(drop=True)
    plt.figure(figsize=(9, 4.4))
    xx = np.arange(len(s))
    plt.fill_between(xx, s.lifetime_lo / 1e4, s.lifetime_hi / 1e4,
                     color="#c4b5fd", alpha=0.6, label="95% 區間")
    plt.plot(xx, s.lifetime_mean / 1e4, color="#6d28d9", lw=1.5, label="平均")
    plt.xlabel(f"病患(n={len(s)},依餘生成本平均排序;重複回診僅取首筆)")
    plt.ylabel(f"餘生醫療成本(萬元,{money_label()})")
    plt.title(f"每位病患餘生成本({money_label()};時界:{horizon_label()})\n"
              f"蒙地卡羅平均與 95% 模擬區間(N={N_SIM}/人)")
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
    plt.xlabel(f"母體餘生總醫療成本(百萬元,{money_label()})"); plt.ylabel("模擬次數")
    plt.title(f"母體餘生總成本的蒙地卡羅分布({money_label()};N={N_SIM} 次)\n"
              f"時界:{horizon_label()};含全體共用之母體參數不確定性")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_population_total.png", dpi=150)
    if _NB: plt.show()
    plt.close()

    # ---- 圖 4:成本集中度勞倫茲曲線 + Gini 係數 ----
    # [修正 2] 集中度以「病患」為單位,不重複計入回診紀錄
    cc = np.sort(out.loc[out["counted_in_population"], "lifetime_mean"].values)
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
    plt.title(f"餘生醫療成本集中度:勞倫茲曲線(n=100 位病患)")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_lorenz.png", dpi=150)
    if _NB: plt.show()
    plt.close()
    print(f"[集中度] Gini={gini:.3f};最貴前 10% 占 {top10:.1f}%、前 20% 占 {top20:.1f}% → mc_lorenz.png")

    # ---- 敏感度分析:類別權重對機率校準與成本的影響 ----
    sensitivity_class_weight(d, targets, pid, age, is_first)

    print(f"\n完成。輸出於 ./{OUTDIR}/  (patient_lifetime_cost_mc.csv、mc_*.png、"
          f"cost_sensitivity_class_weight.csv)")
    print(f"提醒:金額為 {TARGET_YEAR} 年人民幣(上海醫療保健類 CPI 逐年連乘校正)。")
    print("     LIFE_EXPECTANCY 仍為量級估計,建議以官方生命表取代。")


if __name__ == "__main__":
    main()

