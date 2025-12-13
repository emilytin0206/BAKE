# BAKE: Behavioral Alignment & Knowledge Extraction

**BAKE** 是一個自動化的提示詞優化（Prompt Optimization）框架。它不只是單純地尋找「更好的提示詞」，而是透過分析模型在特定任務上的失敗案例，提取出模型偏好的行為規則（Behavioral Alignment），並將這些隱性知識轉化為顯性的指導原則（Knowledge Extraction），最終生成高質量的通用提示詞。

本專案實作了 BAKE 的核心流程，並支援 **迭代式優化（Iterative Optimization）**，允許在優化過程中動態更新提示詞池。

## ✨ 主要功能

  * **雙模型架構**：
      * **Scorer (Task Model)**：負責執行任務並由系統評估對錯（如 Qwen-7B, GPT-3.5）。
      * **Optimizer (Teacher Model)**：負責分析錯誤、重寫提示詞並總結規則（如 Qwen-32B, GPT-4）。
  * **自動化流程**：包含評估 (Evaluation)、優化 (Refinement)、規則提取 (Rule Extraction) 與合併 (Merging)。
  * **迭代模式 (Iterative Mode)**：支援在運行過程中根據新生成的規則即時更新提示詞池，實現「在線學習」。
  * **多模型支援**：透過 `LLMClient` 支援 **OpenAI API** 與 **Ollama** 本地模型。
  * **詳細日誌**：完整記錄優化軌跡、成本估算與規則演變過程。

## 📂 專案結構

```text
BAKE/
├── core/
│   ├── bake_engine.py      # BAKE 核心引擎：處理評估、優化、規則提取邏輯
│   └── llm_client.py       # LLM 客戶端：處理 OpenAI/Ollama 連線與計費
├── utils/
│   ├── config_loader.py    # 讀取 YAML 設定與 Meta Prompts
│   ├── data_loader.py      # 載入資料集 (MMLU, GSM8K)
│   ├── logger.py           # 日誌記錄工具 (Thread-safe)
│   └── text_tools.py       # 文字處理與答案驗證工具
├── meta_prompt/            # [關鍵] 給 Optimizer 使用的元提示詞模板
│   ├── analyze_and_rewrite.txt
│   ├── combine_rules.txt
│   ├── prompt_generation.txt
│   └── rule_summarization.txt
├── config.yaml             # 主要設定檔 (模型參數、路徑、資料集)
├── main.py                 # 程式進入點
├── BAKE.sh                 # 批次實驗自動化腳本
└── requirements.txt        # Python 依賴套件
```

## 🚀 安裝與環境設定

1.  **Clone 專案**

    ```bash
    git clone https://github.com/your-repo/BAKE.git
    cd BAKE
    ```

2.  **安裝依賴套件**
    建議使用 Python 3.10+ 環境：

    ```bash
    pip install -r requirements.txt
    ```

3.  **設定模型後端**

      * **本地模型 (Ollama)**：請確保 Ollama 已啟動並下載了相應模型（如 `qwen2.5:7b`）。
      * **雲端模型 (OpenAI)**：請準備好 API Key。

## ⚙️ 配置說明 (`config.yaml`)

在執行前，請編輯 `config.yaml` 以符合您的環境：

```yaml
# 模型設定
scorer:
  provider: "ollama"       # 或 "openai"
  model_name: "qwen2.5:7b" # 執行任務的小模型
  base_url: "http://localhost:11434/v1"

optimizer:
  provider: "ollama"       # 或 "openai"
  model_name: "qwen2.5:32b" # 負責優化的大模型 (建議能力較強者)

# 資料集設定
datasets:
  - name: "mmlu"
    subsets: ["high_school_mathematics", "professional_law"] # 指定子集
    limit: 10 # 每個子集測試幾題

# 初始提示詞
initial_prompts:
  - "Let's think step by step."
  - "Think about this logically."
```

## 🏃‍♂️ 執行方式

### 方式 1：使用 Python 直接執行

您可以直接執行 `main.py` 並透過參數覆蓋設定檔：

```bash
python main.py \
  --output_dir experiments/run_01 \
  --scorer_model "qwen2.5:7b" \
  --optimizer_model "gpt-4o" \
  --iterative  # 開啟迭代模式 (選填)
```

**參數說明：**

  * `--output_dir`: 輸出結果與 Log 的資料夾路徑（必填）。
  * `--iterative`: 是否開啟迭代模式。若開啟，系統會在生成 Tier-1 規則後立即產生新提示詞並用於後續題目。
  * `--dataset_limit`: 強制覆蓋測試樣本數量。

### 方式 2：使用 Shell 腳本進行批次實驗

專案提供了 `BAKE.sh` 來自動化執行多組實驗配置：

```bash
chmod +x BAKE.sh
./BAKE.sh
```

您可以在 `BAKE.sh` 中修改 `EXPERIMENTS` 陣列來安排實驗佇列：

```bash
EXPERIMENTS=(
    # Scorer | Optimizer | DatasetLimit | IterativeMode
    "qwen2.5:7b|qwen2.5:32b|100|true"
    "gpt-3.5-turbo|gpt-4|50|false"
)
```

## 📊 輸出結果

執行完成後，結果將保存在指定的 `--output_dir` 中：

  * **`optimized_prompts.txt`**：最終生成的優化提示詞列表。
  * **`final_rule.txt`**：最終提取出的通用提示詞設計規則（Behavioral Rule）。
  * **`detailed_results.jsonl`**：每一題的詳細評估結果（包含正確/錯誤的提示詞）。
  * **`optimization_status.csv`**：每一題的優化狀態（成功、失敗、跳過）。
  * **`rule_evolution.jsonl`**：記錄規則從單題屬性到全域規則的演變過程。
  * **`cost_report.csv`**：Token 使用量與預估成本報告。

## 🛠️ 進階客製化

若需調整優化邏輯，請修改 `meta_prompt/` 下的文件：

  * `analyze_and_rewrite.txt`: 指導 Optimizer 如何診斷錯誤並重寫提示詞。
  * `combine_rules.txt`: 指導 Optimizer 如何合併多條規則。
  * `prompt_generation.txt`: 指導 Optimizer 如何根據規則生成新提示詞。

## 📜 引用與參考

本代碼基於 BAKE 論文概念實作。核心邏輯參考自：

> *BAKE: Behavioral Alignment & Knowledge Extraction for Prompt Optimization*

-----

**License**: MIT
**Author**: Emily