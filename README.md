# 糖尿病惡化風險分層、CGM 短期預測與醫療成本情境分析

本專案整合第二型糖尿病臨床摘要資料與連續血糖監測（Continuous Glucose Monitoring, CGM）資料，建立一套包含臨床併發症風險分層、HbA1c 迴歸、非監督式分群、血糖狀態轉移、LSTM 短期預測、二維風險整合，以及醫療成本情境模擬的研究流程。

本 repository 主要包含兩條分析路線：

1. **臨床橫斷面分析**：使用人口學、生活型態與生化檢驗特徵，進行大血管、小血管與任一併發症的關聯分類與風險分層。
2. **CGM 時間序列分析**：計算血糖控制指標、建立 Markov 血糖狀態轉移矩陣，並使用 LSTM 預測下一筆血糖值。

兩條路線後續整合為：

- **風險引擎**：結合臨床併發症風險分層與 CGM 短期血糖失控風險。
- **成本引擎**：結合模型風險與文獻成本參數，進行探索性的醫療成本情境模擬。

---

## 1. 研究定位

臨床摘要資料屬於 **cross-sectional data（橫斷面資料）**。臨床特徵與併發症狀態是在相近時間點記錄，因此臨床分類模型估計的是：

> 病患目前的臨床特徵，與資料中已有大血管或小血管併發症病患之間的相似程度與關聯。

因此，臨床模型應解讀為：

- 目前併發症的關聯分類；
- 病患風險分層；
- 進一步檢查與追蹤的優先排序參考。

臨床模型**不能**解讀為：

- 經過驗證的未來併發症發生率；
- 未來中風、腎病、視網膜病變或神經病變的確定機率；
- 因果關係；
- 臨床診斷。

CGM 序列具有真實時間順序，因此 Markov 狀態轉移與 LSTM 下一步血糖預測才是本專案中真正具有時間性的分析。

本專案僅供研究與教學用途，不可直接作為臨床診斷、治療建議或醫療資源配置工具。

---

## 2. 資料結構

本研究資料包含：

- 109 筆臨床監測紀錄；
- 約 100 位獨立第二型糖尿病病患；
- 部分病患具有多次回診紀錄；
- 109 份 CGM 監測檔；
- CGM 原則上約每 15 分鐘記錄一次血糖。

主要臨床目標：

| Target | 說明 | 本研究定位 |
|---|---|---|
| `Macrovascular` | 是否已有大血管併發症 | 主要分類目標 |
| `Microvascular` | 是否已有小血管併發症 | 主要分類目標 |
| `Any_Complication` | 是否已有任一類併發症 | 綜合風險分層代理指標 |
| `Hypoglycemia` | 是否有低血糖紀錄 | 陽性樣本較少，僅供補充分析 |

原始病患資料、CGM 檔案及含可識別病患資訊的輸出，不應提交至公開 repository。

---

## 3. X 與 Y 的定義

### 3.1 臨床分類模型

#### X：輸入特徵

臨床分類使用人口學、生活型態與生化檢驗資料，例如：

- `Age`
- `Gender`
- `Height`
- `Weight`
- `BMI`
- `Smoking`
- `Alcohol`
- `Duration`
- `FPG`
- `PPG`
- `Cpep_f`
- `Cpep_2h`
- `Ins_f`
- `Ins_2h`
- `HbA1c`
- `GA`
- `TC`
- `TG`
- `HDL`
- `LDL`
- `Cr`
- `eGFR`
- `UA`
- `BUN`

`Patient Number` **不是模型的 X**。它只用來建立病患分組，確保同一位病患不會同時出現在交叉驗證的訓練折與測試折。

用藥與共病變數預設不納入主要 X，因為部分藥物可能是在併發症已發生後才開立，容易形成反向因果或答案洩漏。

#### Y：二元目標

| Target | Y = 0 | Y = 1 |
|---|---|---|
| `Macrovascular` | 無大血管併發症 | 有大血管併發症 |
| `Microvascular` | 無小血管併發症 | 有小血管併發症 |
| `Any_Complication` | 無任何納入定義的併發症 | 至少存在一種併發症 |

程式中的完整定義為：

```text
Any_Complication
= Macrovascular OR Microvascular OR Acute Diabetic Complications
```

由於目前資料中的 `Acute Diabetic Complications` 全為 0，因此本資料實際結果等同於：

```text
Any_Complication
= Macrovascular OR Microvascular
```

這些 Y 代表該次紀錄中**目前是否已有併發症**，不代表未來新發生併發症。

### 3.2 HbA1c 迴歸

- X：其他臨床特徵；
- 排除：`HbA1c` 與 `GA`；
- Y：連續型 `HbA1c` 數值。

排除 `GA` 的原因是其與 HbA1c 高度相關，若納入可能造成近似目標洩漏。

### 3.3 LSTM 血糖預測

- X：過去 12 筆 CGM；
- Y：下一筆 CGM；
- 若資料確實每 15 分鐘一筆，12 筆約代表過去 3 小時；
- 下一筆約代表 15 分鐘後的血糖。

### 3.4 K-means 與 GMM 分群

分群屬非監督式學習：

- 有 X；
- 沒有 Y；
- 併發症標籤只在分群完成後，用於探索不同群的併發症比例是否有差異。

---

## 4. 臨床模型的實務用途

### 對醫師

建議使用以下方式解釋：

> 模型辨識目前臨床特徵與資料中已有大血管或小血管併發症病患較相似的個案，可協助安排進一步檢查、追蹤與臨床評估的優先順序，但不能取代醫師診斷。

模型可作為：

- 初步風險分層；
- 追蹤優先排序；
- 補充檢查提醒；
- 危險因子檢視工具；
- 研究型決策支援資訊。

### 對病患

建議使用以下方式說明：

> 依照您目前的檢驗與健康資料，您的特徵與部分已有相關併發症的病患較相似，因此系統建議提高關注並由醫師進一步評估。這不是確診，也不代表一定會發生併發症。

不應直接向病患表示：

- 「你未來一定會出現併發症」；
- 「你未來有某個百分比會中風」；
- 「模型已經診斷你有腎病或視網膜病變」。

---

## 5. 資料前處理與資料洩漏防治

### 5.1 缺失值處理

原始臨床資料可能以 `/` 表示缺值。程式先將其轉為缺失值，再使用：

```python
SimpleImputer(strategy="median")
```

中位數插補對右偏且可能含離群值的臨床資料較穩健。

### 5.2 對數轉換

以下右偏變數使用 `log1p` 取代原始值：

- `FPG`
- `PPG`
- `Cpep_f`
- `Cpep_2h`
- `Ins_f`
- `Ins_2h`
- `TG`
- `TC`
- `Cr`
- `eGFR`
- `UA`
- `BUN`
- `Duration`

`HbA1c`、`GA`、`HDL`、`LDL` 保留原始尺度。

### 5.3 模型前處理差異

不同模型的 Pipeline 不完全相同：

| 模型 | Pipeline 前處理 |
|---|---|
| Logistic Regression | `SimpleImputer` + `StandardScaler` |
| Linear Regression | `SimpleImputer` + `StandardScaler` |
| XGBoost / HistGradientBoosting 分類 | `SimpleImputer`，不使用 `StandardScaler` |
| XGBoost / HistGradientBoosting 迴歸 | `SimpleImputer`，不使用 `StandardScaler` |

### 5.4 三類資料洩漏

#### 前處理洩漏

問題：若先用全部資料計算中位數、平均值與標準差，再進行交叉驗證，測試資料資訊會提前進入模型。

對策：將插補與標準化放進 `sklearn Pipeline`，使轉換器只在每一折的訓練資料上配適。

#### 病患層級洩漏

問題：同一病患的不同回診紀錄若分別出現在訓練折與測試折，模型等於已經看過該病患。

對策：使用 `GroupKFold`，以 Patient Number 的病患主編號分組。

目前設定：

```python
DEDUP_ONE_PER_PATIENT = False
USE_GROUP_CV = True
```

代表保留全部紀錄，但使用病患分組交叉驗證。

#### 反向因果洩漏

問題：部分藥物或共病欄位可能因併發症已存在才被記錄，若將其作為 X，可能等於把答案提供給模型。

對策：

```python
INCLUDE_MED_FEATURES = False
```

預設排除用藥與共病二元變數。

### 5.5 Out-of-fold 預測

分類、迴歸及風險引擎使用 out-of-fold（OOF）預測。每筆 OOF 預測均來自未使用該筆驗證資料訓練的模型，可降低使用訓練內預測造成的過度樂觀。

---

## 6. Label 與 Mapping

### 6.1 臨床類別編碼

| Variable | Raw value | Model value |
|---|---|---:|
| `Gender` | Female = 1 | 0 |
| `Gender` | Male = 2 | 1 |
| `Smoking` | pack-year = 0 或缺失 | 0 |
| `Smoking` | pack-year > 0 | 1 |
| `Alcohol` | non-drinker | 0 |
| `Alcohol` | drinker | 1 |
| `Hypoglycemia` | no | 0 |
| `Hypoglycemia` | yes | 1 |
| Complication / medication fields | none / no / nan / 空字串 | 0 |
| Complication / medication fields | 有其他紀錄 | 1 |

目前 mapping 分散在各程式中。後續建議建立 `project_schema.py`，作為程式與文件的 single source of truth。

### 6.2 CGM 三狀態

| Code | Label | 血糖範圍 |
|---:|---|---|
| 0 | Low | `< 70 mg/dL` |
| 1 | In Range | `70–180 mg/dL` |
| 2 | High | `> 180 mg/dL` |

### 6.3 CGM 五狀態

| Code | Label | 血糖範圍 |
|---:|---|---|
| 0 | Very Low | `< 54 mg/dL` |
| 1 | Low | `54 <= glucose < 70 mg/dL` |
| 2 | In Range | `70 <= glucose <= 180 mg/dL` |
| 3 | High | `180 < glucose <= 250 mg/dL` |
| 4 | Very High | `> 250 mg/dL` |

---

## 7. 五支主要程式

### 7.1 `code/diabetes_deterioration_pipeline.py`

臨床摘要資料的主要分析管線，包含：

- 臨床 Excel 讀取；
- Patient Number 病患主編號擷取；
- 類別變數二元化；
- 右偏變數 `log1p` 轉換；
- 目標變數建立；
- EDA；
- Logistic Regression 基準模型；
- XGBoost 主要模型；
- 未安裝 XGBoost 時退回 HistGradientBoosting；
- GroupKFold；
- out-of-fold 預測；
- ROC 曲線；
- 混淆矩陣；
- SHAP；
- permutation importance；
- Logistic Regression odds ratio；
- K-means；
- GMM；
- silhouette score；
- BIC；
- PCA 視覺化；
- 分群後卡方檢定。

重要設定：

```python
SEED = 42
XLSX_PATH = "Shanghai_T2DM_Summary.xlsx"
OUTDIR = "output"
DEDUP_ONE_PER_PATIENT = False
USE_GROUP_CV = True
INCLUDE_MED_FEATURES = False
RUN_EDA = True
```

特徵重要性輸出說明：

- `IMP_{target}.png` 會依重要性表產生；
- 若 XGBoost 與 SHAP 均可使用，會另外產生 `SHAP_{target}.png`；
- SHAP 並不是 `IMP` 圖的替代檔，兩者可能同時存在。

### 7.2 `code/diabetes_regression.py`

以臨床特徵預測 HbA1c：

- Y：`HbA1c`；
- X：排除 `HbA1c` 與 `GA` 的其餘臨床特徵；
- Linear Regression 為基準；
- XGBoost Regressor 為主要模型；
- 未安裝 XGBoost 時退回 HistGradientBoostingRegressor；
- GroupKFold；
- out-of-fold 預測；
- MAE；
- RMSE；
- MAPE；
- R²；
- actual vs predicted 圖；
- residual plot；
- permutation importance。

重要設定：

```python
SEED = 42
OUTDIR = "output_reg"
TARGET = "HbA1c"
EXCLUDE_AS_FEATURE = ["HbA1c", "GA"]
```

### 7.3 `code/cgm_lstm_markov.py`

CGM 分析內容：

- 讀取 `.xls` 與 `.xlsx`；
- 擷取 timestamp 與 CGM 欄位；
- 計算 Mean、SD、CV、GMI、TIR、TBR、TAR；
- 建立合併全部 CGM 檔案的母體 3-state Markov matrix；
- 建立合併全部 CGM 檔案的母體 5-state Markov matrix；
- 計算 stationary distribution；
- 繪製 individual fluctuation plot；
- 繪製 pooled AGP；
- 繪製 12 份監測的示範網格；
- 繪製 CGM 指標分布；
- 單一病患 LSTM；
- 全體監測 persistence baseline；
- 全體監測逐一訓練 LSTM 並與 persistence baseline 比較。

重要設定：

```python
SEED = 42
CGM_DIR = "Shanghai_T2DM"
OUTDIR = "cgm_output"
```

#### Markov 時間間隔限制

目前母體 Markov 函式使用：

```python
markov_matrix(n_states=3, max_gap_min=30)
```

這代表兩筆資料時間差只要不超過 30 分鐘，就可能被計入一次狀態轉移。這並不是嚴格的 15 分鐘轉移。

正式版本建議只保留：

```python
14 <= gap_minutes <= 16
```

#### LSTM 設定與限制

預設：

```python
look_back = 12
horizon = 1
```

資料先做前 80% 訓練、後 20% 測試，標準化的平均值與標準差只使用訓練段計算。

但目前仍有兩項限制：

1. `make_windows()` 依序列索引建立視窗，未檢查 timestamp，因此視窗可能跨越缺測區段。
2. `model.fit()` 使用 `validation_split=0.15`，未明確設定 `shuffle=False`，因此不能宣稱內部 validation 完全保持時間順序。

### 7.4 `code/deterioration_risk.py`

整合兩個風險軸。

#### 長期軸

```text
Logistic Regression 對 Any_Complication 的 GroupKFold OOF 機率
```

此分數代表橫斷面併發症風險分層，不是經縱向驗證的未來發生率。

#### 短期軸

使用每一份 CGM 的 3-state Markov matrix，從 `InRange` 狀態推算：

- 1 小時：4 步；
- 4 小時：16 步；
- 24 小時：96 步。

短期分數為：

```text
hypo_24h + hyper_24h
```

#### 四象限

| 象限 | 定義 | 程式標籤 |
|---|---|---|
| A | 長期高、短期高 | 立即介入 |
| B | 長期高、短期低 | 慢性追蹤 |
| C | 長期低、短期高 | 血糖波動注意 |
| D | 長期低、短期低 | 常規追蹤 |

象限以樣本中位數切分，屬探索性分層，不是已驗證的臨床門檻。

#### 已知時間限制

`transition_matrix(states, max_gap_steps=2, ts=None)` 雖接受時間相關參數，但目前沒有實際使用 timestamp 篩選轉移，因此可能把跨缺測區段的兩筆資料當成相鄰轉移。

### 7.5 `code/lifetime_cost_montecarlo.py`

目前程式使用：

- Macrovascular OOF probability；
- Microvascular OOF probability；
- Bernoulli complication-state sampling；
- Gamma cost uncertainty；
- Gamma inflation-factor uncertainty；
- 折現年金；
- Monte Carlo PSA；
- 平均值；
- 2.5% 與 97.5% 百分位數。

重要設定：

```python
SEED = 42
N_SIM = 5000
COST_TYPE = "direct"
COST_CV = 0.30
INFLATION_MEAN = 1.8
INFLATION_CV = 0.15
DISCOUNT_RATE = 0.03
HORIZON_AGE = 80
OUTDIR = "output_cost_mc"
```

#### 成本來源與幣值

程式註記的成本來源為陳興寶等於 2003 年發表的研究，但表中的成本數值是：

```text
2001 年人民幣（RMB）
```

因此物價轉換的正確起始年應是 2001 年，而不是 2003 年。

目前：

```python
INFLATION_MEAN = 1.8
```

只是 placeholder，不是官方上海醫療保健 CPI 的正式轉換因子。正式分析應使用上海市醫療保健類居民消費價格指數，逐年鏈結 2001 至 2025 年。

在官方轉換率與其他成本假設完成前，本程式應稱為：

> 探索性醫療成本情境分析。

不應稱為已驗證的個人餘生醫療成本預測。

#### 成本模型限制

- Macro 與 Micro 狀態分開抽樣，隱含給定 X 後條件獨立的假設；
- 每筆臨床紀錄均進入成本迴圈，重複回診者可能被重複計入母體總額；
- 沒有逐年併發症狀態轉移；
- 沒有死亡競爭風險；
- 目前風險被外推至未來整個時界；
- 不是完整 disease microsimulation；
- 比較穩健的近期替代方案，是估計 2025 年年度成本或固定五年情境成本。

---

## 8. 模型與評估指標

| Task | Baseline | Main model | Metrics |
|---|---|---|---|
| Classification | Logistic Regression | XGBoost / HistGradientBoosting | AUC、AP、Accuracy、Precision、Recall、F1 |
| Regression | Linear Regression | XGBoost Regressor / HistGradientBoostingRegressor | MAE、RMSE、MAPE、R² |
| Clustering | K-means | GMM | Silhouette、BIC、探索性卡方檢定 |
| Time series | Persistence baseline | LSTM、Markov | RMSE、MAE、Skill |
| Cost | 文獻成本情境 | Monte Carlo PSA | Mean、2.5% 與 97.5% 百分位數 |

README 不直接列入未經目前 repository 輸出驗證的模型數值。

---

## 9. Repository Structure

目前 repository 的規劃結構：

```text
Diabetes_Deterioration_Risk_Project/
├── code/
│   ├── diabetes_deterioration_pipeline.py
│   ├── diabetes_regression.py
│   ├── cgm_lstm_markov.py
│   ├── deterioration_risk.py
│   └── lifetime_cost_montecarlo.py
├── data/
│   ├── clinical/
│   └── cgm/
├── output/
│   ├── clinical/
│   ├── regression/
│   ├── cgm/
│   ├── risk/
│   └── cost/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

但是，五支程式目前仍使用舊版相對路徑，尚未自動寫入上述分類後的 `output/` 子目錄。實際路徑請參考下方說明。

---

## 10. 安裝方式

建議使用 Python 3.10 以上版本。

建立虛擬環境：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

建議先更新 pip，再安裝專案依賴：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

套件用途：

| Package | 用途 |
|---|---|
| `numpy` | 陣列與數值計算 |
| `pandas` | Excel、表格與資料處理 |
| `scipy` | 卡方檢定 |
| `scikit-learn` | Pipeline、分類、迴歸、分群與評估 |
| `matplotlib` | 圖表輸出 |
| `seaborn` | EDA heatmap |
| `openpyxl` | 讀取 `.xlsx` |
| `xlrd` | 讀取 `.xls` |
| `xgboost` | XGBoost 分類與迴歸 |
| `shap` | SHAP 模型解釋 |
| `tensorflow` | LSTM |

---

## 11. 資料放置與路徑限制

### 11.1 Repository 規劃位置

```text
data/clinical/Shanghai_T2DM_Summary.xlsx
data/cgm/*.xls
data/cgm/*.xlsx
```

### 11.2 現行程式實際期待位置

目前程式中的常數為：

```text
Shanghai_T2DM_Summary.xlsx
Shanghai_T2DM/
output/
output_reg/
cgm_output/
output_risk/
output_cost_mc/
```

也就是說，直接執行程式時，臨床 Excel 預設從目前工作目錄讀取：

```text
Shanghai_T2DM_Summary.xlsx
```

CGM 預設從目前工作目錄下的：

```text
Shanghai_T2DM/
```

讀取。

若資料已放在 `data/clinical/` 與 `data/cgm/`，必須先修改程式常數，或將資料放到程式目前預期的位置。

---

## 12. 執行順序

在 repository 根目錄執行：

```bash
python code/diabetes_deterioration_pipeline.py
python code/diabetes_regression.py
python code/cgm_lstm_markov.py
python code/deterioration_risk.py
python code/lifetime_cost_montecarlo.py
```

依賴關係：

- `diabetes_regression.py` 會 import `load_data()`；
- `deterioration_risk.py` 會 import `load_data()`；
- `lifetime_cost_montecarlo.py` 會 import `load_data()`；
- 三者均依賴 `diabetes_deterioration_pipeline.py`；
- `deterioration_risk.py` 同時需要臨床 Excel 與 CGM；
- 成本程式若找到 `output_risk/patient_risk_table.csv`，會嘗試合併風險象限。

---

## 13. 實際輸出檔案

### 13.1 臨床分類、EDA 與分群

輸出目錄：

```text
output/
```

主要輸出：

```text
model_results.csv
ROC_Any_Complication.png
ROC_Microvascular.png
ROC_Macrovascular.png
CM_Any_Complication.png
CM_Microvascular.png
CM_Macrovascular.png
IMP_Any_Complication.png
IMP_Microvascular.png
IMP_Macrovascular.png
importance_Any_Complication.csv
importance_Microvascular.csv
importance_Macrovascular.csv
lr_oddsratio_Any_Complication.csv
lr_oddsratio_Microvascular.csv
lr_oddsratio_Macrovascular.csv
Clustering_PCA.png
```

若 XGBoost 與 SHAP 可用，另外產生：

```text
SHAP_Any_Complication.png
SHAP_Microvascular.png
SHAP_Macrovascular.png
```

EDA：

```text
output/eda_figures/
├── 00_target_distribution.png
├── 01_missing_heatmap.png
├── 02_histograms.png
├── 03_boxplots.png
├── 04_correlation_matrix.png
├── missing_pct.csv
├── correlation_matrix.csv
└── describe.csv
```

### 13.2 HbA1c 迴歸

輸出目錄：

```text
output_reg/
```

輸出：

```text
reg_results.csv
reg_pred_vs_actual_HbA1c.png
reg_residuals_HbA1c.png
reg_importance_HbA1c.csv
reg_importance_HbA1c.png
```

### 13.3 CGM、Markov 與 LSTM

輸出目錄：

```text
cgm_output/
```

固定或主流程輸出：

```text
cgm_metrics.csv
markov_3state.csv
markov_3state.png
markov_5state.csv
markov_5state.png
AGP_all.png
fluctuation_grid12.png
cgm_metrics_dist.png
cgm_descriptive.csv
fluctuation_<record>.png
baseline_all.csv
```

若 TensorFlow 可使用且指定病患資料存在：

```text
lstm_forecast_2000.png
```

若全體 LSTM 執行成功：

```text
lstm_vs_baseline_all.csv
lstm_vs_baseline_all.png
```

### 13.4 風險引擎

輸出目錄：

```text
output_risk/
```

輸出：

```text
patient_risk_table.csv
risk_map_2d.png
quadrant_summary.csv
```

### 13.5 成本情境模擬

輸出目錄：

```text
output_cost_mc/
```

固定輸出：

```text
patient_lifetime_cost_mc.csv
mc_cost_per_patient.png
mc_population_total.png
mc_lorenz.png
```

若成功合併含 `quadrant` 欄位的風險表，另外產生：

```text
mc_cost_by_quadrant.png
```

---

## 14. 重現性

目前已實作：

- `SEED = 42`；
- Logistic Regression、XGBoost、HistGradientBoosting、K-means、PCA 相關流程固定亂數設定；
- 分類與迴歸使用 Pipeline；
- 分類與迴歸使用病患分組交叉驗證；
- 評估使用 OOF 預測；
- LSTM train/test 使用時間順序切分；
- CGM 標準化只使用訓練段統計量；
- TensorFlow 使用固定 seed；
- Monte Carlo 使用固定 RNG seed。

仍須注意：

- 部分深度學習運算在不同硬體與 TensorFlow 環境下未必完全 deterministic；
- LSTM 的內部 validation 尚未明確設為 `shuffle=False`；
- 分群的 permutation importance 是在全資料 fit 後計算，屬探索性解釋，不等同外部驗證。

---

## 15. 研究限制

1. 臨床資料是橫斷面資料，不能直接推論未來發病或因果關係。
2. 樣本約 100 位病患，模型效能與特徵重要性可能具有較大不確定性。
3. 部分病患有重複紀錄，必須使用 GroupKFold 或病患層級去重。
4. 部分生化變數缺失比例較高，中位數插補仍可能造成偏差。
5. 分群為探索性分析，不代表已驗證的臨床亞型。
6. 四象限以樣本中位數切分，不是臨床標準門檻。
7. 母體 AGP 與 Markov matrix 以監測事件合併，監測時間較長者可能具有較高權重。
8. 母體 Markov 目前接受不超過 30 分鐘的轉移，尚未嚴格限制為 14–16 分鐘。
9. 風險引擎的個人 Markov matrix 目前沒有使用 timestamp 排除跨缺測轉移。
10. LSTM 視窗可能跨越時間缺口。
11. LSTM 內部 validation 未明確設定 `shuffle=False`。
12. 單一病患 LSTM 結果不能直接推論至整體族群。
13. 尚未使用外部資料進行驗證。
14. `requirements.txt` 已列出主要依賴套件，但尚未鎖定精確版本，也尚未在全新的乾淨環境完成完整重現測試。
15. 輸入與輸出路徑仍為 legacy relative paths。
16. 成本 CPI 倍數仍為 placeholder。
17. 成本模型可能重複計算同一病患的多次紀錄。
18. 成本模型假設 Macro 與 Micro 在給定 X 後可分開抽樣。
19. 成本模型沒有逐年病程進展與死亡模型。
20. 成本輸出目前只能視為探索性情境分析。

---

## 16. 資料隱私

本 repository 不包含：

- 原始病患臨床資料；
- 原始 CGM 檔案；
- 可識別病患的輸出表；
- 任何應受保護的個人健康資訊。

請勿提交：

```text
Shanghai_T2DM_Summary.xlsx
Shanghai_T2DM/*.xls
Shanghai_T2DM/*.xlsx
data/clinical/*.xls
data/clinical/*.xlsx
data/cgm/*.xls
data/cgm/*.xlsx
```

即使 `.gitignore` 已排除部分檔案，提交前仍應人工確認：

- CSV 是否含 Patient Number；
- 圖表標題是否含病患識別碼；
- 輸出資料是否可能重新識別個人；
- repository 是否設定為適當的公開或私人權限。

---

## 17. 後續改進方向

- 建立 `project_schema.py`，集中管理 label、mapping、狀態門檻與成本 metadata；
- 建立 `generate_documentation.py`，自動產生變數代碼表；
- 統一 `data/` 與 `output/` 路徑；
- 在乾淨的虛擬環境完成依賴安裝與五支程式的執行測試，並建立可重現的版本鎖定檔；
- Markov 僅保留 14–16 分鐘有效轉移；
- LSTM 在缺口處切段，且使用 chronological validation 或 `shuffle=False`；
- 分群前每位病患只保留一筆基準紀錄；
- 風險表與成本表改為每位病患一列；
- 以官方上海醫療保健 CPI 逐年鏈結 2001–2025；
- 將成本分析優先改成 2025 年年度成本或固定五年情境；
- 若要建立真正的未來併發症與餘生成本模型，需加入縱向事件資料、逐年轉移率與死亡競爭風險。
