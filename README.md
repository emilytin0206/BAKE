# BAKE: 自動化提示詞優化框架 (Automated Prompt Optimization Framework)

BAKE 是一個自動化的提示詞（Prompt）優化工具，旨在通過錯誤分析與迭代改進來提升大型語言模型（LLM）在特定任務上的表現。

本框架利用「評估者（Scorer）」與「優化者（Optimizer）」雙模型架構：評估者負責回答問題並驗證對錯，優化者則根據錯誤案例分析原因、重寫提示詞並歸納出通用的解題規則。

## 核心功能 (Features)

* **自動化優化流程**：
    * **評估 (Evaluate)**：在訓練集上並行測試多個提示詞。
    * **改進 (Refine)**：針對錯誤案例，讓優化模型分析並生成修正後的提示詞。
    * **驗證 (Verify)**：立即驗證修正後的提示詞是否有效。
    * **規則提取 (Rule Extraction)**：從成功的改進中提取通用規則，並進行分層合併 (Recursive Merge)。
* **多任務支援**：內建支援 `MMLU` 與 `GSM8K` 資料集，並可針對 MMLU 指定特定子集 (Subsets)。
* **靈活的模型後端**：支援 OpenAI API 與 Ollama 本地模型 (如 Qwen2.5)。
* **詳細的實驗記錄**：自動記錄成本 (Cost)、優化狀態、Prompt 演變歷史與詳細的規則推導過程。
* **實驗管理**：提供 Shell 腳本自動根據參數生成規範化的實驗目錄名稱。

## 安裝 (Installation)

請確保您的環境已安裝 Python 3.10+，並安裝以下依賴套件：

```bash
pip install -r requirements.txt

```

主要的依賴包括：

* `openai` (用於串接 LLM API)
* `datasets` (載入 HuggingFace 資料集)
* `PyYAML` (設定檔處理)
* `numpy`, `tqdm` (工具庫)

## 設定 (Configuration)

主要的設定檔位於 `config.yaml`。您可以修改此檔案來調整模型參數、路徑或資料集設定。

### `config.yaml` 關鍵欄位說明：

* **dataset**:
* `active_task`: 設定執行的任務 (如 "mmlu" 或 "gsm8k")。
* `limit`: 設定實驗用的資料筆數。
* `split`: 指定使用的資料集分割 (如 "test" 或 "train")。


* **evaluation**:
* 設定負責解題的評估模型 (Scorer)，例如 `qwen2.5:7b`。
* 支援設定 `provider` (ollama/openai) 與 `pricing` (用於計算成本)。


* **optimizer**:
* 設定負責分析與改進的優化模型 (Optimizer)。
* 建議使用參數量較大、邏輯較強的模型 (如 `qwen2.5:32b`) 以獲得更好的分析效果。


* **bake**:
* `iterative`: 是否開啟迭代模式 (基於歸納出的規則生成新 Prompt 繼續優化)。
* `group_size`: 規則合併時的分組大小。



## 使用方法 (Usage)

### 1. 使用自動化腳本 (推薦)

`BAKE.sh` 是主要的啟動腳本，它會自動處理參數並根據設定生成帶有時間戳記與實驗參數的輸出目錄名稱，方便管理多次實驗。

**步驟：**

1. 編輯 `BAKE.sh` 中的變數以符合您的實驗需求：
```bash
TASK="mmlu"              # 任務類型
SUBSETS="all"            # MMLU 子集 (如 "high_school_mathematics")，"all" 代表全部
LIMIT=100                # 訓練資料筆數 (-1 為全部)
ITERATIVE="true"         # 是否開啟迭代優化
EVAL_MODEL="qwen2.5:7b"  # 評估模型名稱 (Scorer)
OPT_MODEL="qwen2.5:32b"  # 優化模型名稱 (Optimizer)

```


2. 執行腳本：
```bash
chmod +x BAKE.sh
./BAKE.sh

```



實驗結果將會自動儲存於 `experiments/BAKE_<參數組合>_<日期時間>` 目錄下。

### 2. 直接執行 Python 主程式

您也可以直接透過 `main.py` 執行，這在需要手動除錯或整合時很有用：

```bash
python main.py \
  --output_dir experiments/my_custom_run \
  --task mmlu \
  --subsets high_school_mathematics \
  --limit 50 \
  --eval_model qwen2.5:7b \
  --opt_model qwen2.5:32b \
  --iterative

```

**參數說明**：

* `--output_dir`: **(必填)** 輸出目錄路徑。
* `--task`: 指定任務 (mmlu/gsm8k)。
* `--subsets`: 指定子集 (僅 MMLU 有效)。
* `--eval_model` / `--scorer_model`: 覆寫 Config 中的評估模型名稱。
* `--opt_model` / `--optimizer_model`: 覆寫 Config 中的優化模型名稱。
* `--limit` / `--dataset_limit`: 設定使用的資料筆數。
* `--iterative`: 開啟迭代優化模式。

## 輸出檔案說明 (Outputs)

實驗完成後，指定的 `output_dir` 中會包含以下重要產出檔案，您可直接取用這些結果進行下游任務：

* **`final_rule.txt`**: 最終歸納出的通用 Prompting 規則 (Guideline)。
* **`optimized_prompts.txt`**: 基於最終規則生成的優化後 Prompt 列表。
* **`experiment_config.yaml`**: 實驗當下的完整設定檔快照 (Snapshot)。
* **`logs/` 目錄**：
* `optimization_status.csv`: 每一筆資料的優化狀態 (Success/Failed/Skipped)。
* `rules_history.txt`: 詳細記錄從單題規則到最終規則的合併過程。
* `refinement_trace.jsonl`: 記錄錯誤 Prompt -> 修正 Prompt -> 驗證結果的追蹤紀錄。
* `detailed_results.jsonl`: 每輪評估的詳細輸出 (包含 Correct/Wrong 列表)。
* `cost_report.csv`: Token 使用量與預估成本報告。



```

```