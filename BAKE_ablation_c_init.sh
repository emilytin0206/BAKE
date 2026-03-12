#!/bin/bash
# ==========================================
# 🧪 BAKE Ablation C: Success-Only Experiment
# ==========================================

TASK="mmlu"
SUBSETS="high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics" 
SPLIT="test"
LIMIT=100
ITERATIVE="true"
ITER_COUNT=5
SHUFFLE="false" 

EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"


# =========== 👇Initial Prompt👇 ===========
INIT_SRC="gemini"  # 您可以在這裡改成 "gemini"

echo "############################################################"
echo "▶️  Starting Ablation C for Source: $INIT_SRC"
echo "############################################################"

# 呼叫 update_config.py 來修改 config.yaml
echo "🔄 Updating config.yaml with prompts from: $INIT_SRC"
python3 update_config.py "$INIT_SRC"

if [ $? -ne 0 ]; then
    echo "❌ Failed to update config for $INIT_SRC. Exiting..."
    exit 1
fi
# ==========================================

# --- 自動標籤生成邏輯 (Auto Labeling) ---

# 1. Model Name Safe
T_MODEL_SAFE=${EVAL_MODEL//:/-}
O_MODEL_SAFE=${OPT_MODEL//:/-}

# 2. Dataset Label (DS_LABEL)
if [ "$TASK" == "mmlu" ]; then DS_LABEL="MMLU"; else DS_LABEL="GSM8K"; fi

# 3. Subset Label (SUB_LABEL)
if [ "$SUBSETS" == "all" ]; then
    SUB_LABEL="All"
else
    # 計算逗號數量+1來得知有幾個子集
    CNT=$(echo $SUBSETS | tr -cd ',' | wc -c)
    CNT=$((CNT+1))
    SUB_LABEL="${CNT}Sub"
fi

# 4. Limit Label (LIM_LABEL)
LIM_LABEL="Lim${LIMIT}"

# 5. Mode & Count Label (MODE_LABEL, COUNT_LABEL)
if [ "$ITERATIVE" == "true" ]; then
    MODE_LABEL="Iter"
    COUNT_LABEL="${ITER_COUNT}"
else
    MODE_LABEL="Base"
    COUNT_LABEL="0"
fi

# 6. Shuffle Label (SHUFFLE_LABEL)
if [ "$SHUFFLE" == "true" ]; then SHUFFLE_LABEL="Shuffle"; else SHUFFLE_LABEL="Seq"; fi



# 為了讓檔名區分來源，新增 INIT_LABEL
if [ "$INIT_SRC" == "gpt4o" ]; then 
    INIT_LABEL="InitGpt4o"
else 
    INIT_LABEL="InitGemini"
fi


# 7. Date Label
DATE_LABEL=$(date +"%Y%m%d-%H%M%S")

# --- 定義 EXP_NAME (Ablation C) ---
EXP_NAME="BAKE_AblationC_${INIT_LABEL}_${T_MODEL_SAFE}_${O_MODEL_SAFE}_${DS_LABEL}_${SUB_LABEL}_${LIM_LABEL}_${MODE_LABEL}_${COUNT_LABEL}_${SHUFFLE_LABEL}_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

echo "========================================"
echo "🚀 Starting Ablation C (Success-Only Mode)"
echo "📂 Output: $OUTPUT_DIR"
echo "🔀 Mode: $MODE_LABEL | Count: $COUNT_LABEL | Shuffle: $SHUFFLE_LABEL"
echo "========================================"

CMD="python run_ablation_c.py \
    --output_dir $OUTPUT_DIR \
    --task $TASK \
    --subsets $SUBSETS \
    --limit $LIMIT \
    --split $SPLIT \
    --eval_model $EVAL_MODEL \
    --opt_model $OPT_MODEL \
    --iterative_count $ITER_COUNT"

if [ "$ITERATIVE" == "true" ]; then
    CMD="$CMD --iterative"
fi

if [ "$SHUFFLE" == "true" ]; then
    CMD="$CMD --shuffle"
fi

echo "Running: $CMD"
$CMD