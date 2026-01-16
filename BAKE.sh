#!/bin/bash
# ==========================================
# 🧪 BAKE Experiment Runner (Auto-Naming v3)
# ==========================================

# 1. 參數設定 (Settings)
# ------------------------------------------
TASK="mmlu"                       # "mmlu" 或 "gsm8k"
# SUBSETS="all"
SUBSETS="high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics" 
                                  # 若為 "all" 代表全部，否則用逗號分隔
SPLIT="test"
LIMIT=100                         # -1 代表全部
ITERATIVE="false"                  # "true" 開啟迭代, "false" 關閉
ITER_COUNT=5                      # 迭代產生的 Prompt 數量

# [New] 實驗模式設定
SHUFFLE="false"   # => "true": 混合打散模式 / "false": 原始順序

# 模型設定 (注意: 腳本會自動將冒號 ':' 轉為 '-')
EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"

# ==========================================
# 🧠 Auto-Naming Logic (Strict Format with Prefix)
# ==========================================
# 格式: BAKE_<target>_<opt>_<dataset>_<subset>_<limit>_<iter>_<iter_count>_<shuffle>_<date>

# 1. 處理模型名稱 (移除冒號)
T_MODEL_SAFE=${EVAL_MODEL//:/-}
O_MODEL_SAFE=${OPT_MODEL//:/-}

# 2. 處理 Dataset 與 Subset
if [ "$TASK" == "mmlu" ]; then
    DS_LABEL="MMLU"
    if [ "$SUBSETS" == "all" ]; then
        SUB_LABEL="All"
    else
        # 計算逗號分隔的子集數量
        IFS=',' read -ra ADDR <<< "$SUBSETS"
        COUNT=${#ADDR[@]}
        SUB_LABEL="${COUNT}Sub"
    fi
else
    DS_LABEL="${TASK^^}"  # 轉大寫 (GSM8K)
    SUB_LABEL="NA"        # GSM8K 沒有 subset
fi

# 3. 處理 Limit
LIM_LABEL="Lim${LIMIT}"

# 4. 處理 Iter 與 Count
if [ "$ITERATIVE" == "true" ]; then
    MODE_LABEL="Iter"
    COUNT_LABEL="${ITER_COUNT}"
else
    MODE_LABEL="Base"
    COUNT_LABEL="0"
fi

# 5. [New] 處理 Shuffle 標記
if [ "$SHUFFLE" == "true" ]; then
    SHUFFLE_LABEL="Shuffle"
else
    SHUFFLE_LABEL="Seq" # Sequential (原始順序)
fi

# 6. 取得時間
DATE_LABEL=$(date +"%Y%m%d-%H%M%S")

# 7. 組合最終名稱 (加上 BAKE 前綴 與 Shuffle 標籤)
EXP_NAME="BAKE_${T_MODEL_SAFE}_${O_MODEL_SAFE}_${DS_LABEL}_${SUB_LABEL}_${LIM_LABEL}_${MODE_LABEL}_${COUNT_LABEL}_${SHUFFLE_LABEL}_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

# ==========================================
# 🚀 Execution
# ==========================================

echo "========================================"
echo "🔥 Starting Experiment"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "----------------------------------------"
echo "📊 Format: BAKE_<target>_<opt>_<dataset>_<subset>_<limit>_<iter>_<count>_<shuffle>_<date>"
echo "👉 Generated: $EXP_NAME"
echo "🔀 Shuffle:   $SHUFFLE"
echo "========================================"

# 建構指令
CMD="python main.py --output_dir $OUTPUT_DIR --task $TASK --limit $LIMIT --split $SPLIT"

if [ "$TASK" == "mmlu" ]; then
    CMD="$CMD --subsets $SUBSETS"
fi

if [ ! -z "$EVAL_MODEL" ]; then
    CMD="$CMD --eval_model $EVAL_MODEL"
fi

if [ ! -z "$OPT_MODEL" ]; then
    CMD="$CMD --opt_model $OPT_MODEL"
fi

if [ "$ITERATIVE" == "true" ]; then
    CMD="$CMD --iterative --iterative_count $ITER_COUNT"
fi

# [New] 傳遞 Shuffle 參數給 Python
if [ "$SHUFFLE" == "true" ]; then
    CMD="$CMD --shuffle"
fi

# 執行
echo "Running command: $CMD"
$CMD

# 檢查結果
if [ $? -eq 0 ]; then
    echo "✅ Done! Results saved in $OUTPUT_DIR"
else
    echo "❌ Experiment Failed."
fi