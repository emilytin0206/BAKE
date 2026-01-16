#!/bin/bash
# ==========================================
# 🧪 BAKE Experiment Runner (Ablation Study)
# ==========================================

# 1. 參數設定 (Settings)
# ------------------------------------------
TASK="mmlu"                       # "mmlu" 或 "gsm8k"
SUBSETS="high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics" 
SPLIT="test"
LIMIT=100                         # -1 代表全部
ITERATIVE="true"                 # "true" 開啟迭代, "false" 關閉
ITER_COUNT=5                      # 迭代產生的 Prompt 數量

# 實驗模式設定
SHUFFLE="true"   # => "true": 混合打散模式 / "false": 原始順序 (Seq)

# 模型設定
EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"

# ==========================================
# 🧠 Auto-Naming Logic (BAKE_AblationC Format)
# ==========================================
# 目標格式: BAKE_AblationC_<target>_<opt>_<dataset>_<subset>_<limit>_<shuffle>_<count>_<date>

# 1. 處理模型名稱 (移除冒號)
T_MODEL_SAFE=${EVAL_MODEL//:/-}
O_MODEL_SAFE=${OPT_MODEL//:/-}

# 2. 處理 Dataset 與 Subset
if [ "$TASK" == "mmlu" ]; then
    DS_LABEL="MMLU"
    if [ "$SUBSETS" == "all" ]; then
        SUB_LABEL="All"
    else
        IFS=',' read -ra ADDR <<< "$SUBSETS"
        COUNT=${#ADDR[@]}
        SUB_LABEL="${COUNT}Sub"
    fi
else
    DS_LABEL="${TASK^^}"
    SUB_LABEL="NA"
fi

# 3. 處理 Limit
LIM_LABEL="Lim${LIMIT}"

# 4. 處理 Count (若不開啟迭代顯示 0，開啟則顯示數量)
if [ "$ITERATIVE" == "true" ]; then
    COUNT_LABEL="${ITER_COUNT}"
else
    COUNT_LABEL="0"
fi

# 5. 處理 Shuffle 標記
if [ "$SHUFFLE" == "true" ]; then
    SHUFFLE_LABEL="Shuffle"
else
    SHUFFLE_LABEL="Seq"
fi

# 6. 取得時間
DATE_LABEL=$(date +"%Y%m%d-%H%M%S")

# 7. 組合最終名稱 (加上 BAKE_AblationC 前綴)
EXP_NAME="BAKE_AblationC_${T_MODEL_SAFE}_${O_MODEL_SAFE}_${DS_LABEL}_${SUB_LABEL}_${LIM_LABEL}_${SHUFFLE_LABEL}_${COUNT_LABEL}_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

# ==========================================
# 🚀 Execution
# ==========================================

echo "========================================"
echo "🔥 Starting Ablation Study"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "----------------------------------------"
echo "📊 Format: BAKE_AblationC_Target_Opt_DS_Sub_Lim_Shuffle_Count_Date"
echo "👉 Generated: $EXP_NAME"
echo "🔀 Shuffle:   $SHUFFLE"
echo "🔄 Iterative: $ITERATIVE"
echo "========================================"

# [重要修正] 指向 run_ablation_c.py
CMD="python run_ablation_c.py --output_dir $OUTPUT_DIR --task $TASK --limit $LIMIT --split $SPLIT"

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