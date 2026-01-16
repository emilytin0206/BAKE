#!/bin/bash
# ==========================================
# 🧪 BAKE Ablation B: Concise Rule Experiment
# ==========================================

TASK="mmlu"
SUBSETS="high_school_mathematics"  # 或其他子集
LIMIT=10
ITERATIVE="true"
ITER_COUNT=5

EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"

# 自動命名
T_MODEL_SAFE=${EVAL_MODEL//:/-}
O_MODEL_SAFE=${OPT_MODEL//:/-}#!/bin/bash
# ==========================================
# 🧪 BAKE Ablation B: Concise Rule Experiment
# ==========================================

TASK="mmlu"
SUBSETS="high_school_mathematics"  
LIMIT=10
SPLIT="test"          # <--- [新增]
SHUFFLE="false"       # <--- [新增] "true" 開啟打散, "false" 依序
ITERATIVE="true"
ITER_COUNT=5

EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"

# 自動命名
T_MODEL_SAFE=${EVAL_MODEL//:/-}
O_MODEL_SAFE=${OPT_MODEL//:/-}
DATE_LABEL=$(date +"%Y%m%d-%H%M%S")

# 處理 Shuffle 標籤
if [ "$SHUFFLE" == "true" ]; then
    SHUFFLE_LABEL="Shuffle"
else
    SHUFFLE_LABEL="Seq"
fi

# 名稱加上 "_Concise" 以示區別
EXP_NAME="BAKE_AblationB_Concise_${T_MODEL_SAFE}_${O_MODEL_SAFE}_Lim${LIMIT}_${SHUFFLE_LABEL}_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

echo "========================================"
echo "🚀 Starting Ablation B (Concise Mode)"
echo "📂 Output: $OUTPUT_DIR"
echo "🔀 Shuffle: $SHUFFLE | Split: $SPLIT"
echo "========================================"

# 建構指令
CMD="python run_ablation_b.py \
    --output_dir $OUTPUT_DIR \
    --task $TASK \
    --subsets $SUBSETS \
    --limit $LIMIT \
    --split $SPLIT \
    --eval_model $EVAL_MODEL \
    --opt_model $OPT_MODEL \
    --iterative \
    --iterative_count $ITER_COUNT"

# 如果 SHUFFLE 為 true，加入 flag
if [ "$SHUFFLE" == "true" ]; then
    CMD="$CMD --shuffle"
fi

# 執行
echo "Running: $CMD"
$CMD
DATE_LABEL=$(date +"%Y%m%d-%H%M%S")

# 名稱加上 "_Concise" 以示區別
EXP_NAME="BAKE_AblationB_Concise_${T_MODEL_SAFE}_${O_MODEL_SAFE}_Lim${LIMIT}_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

echo "========================================"
echo "🚀 Starting Ablation B (Concise Mode)"
echo "📂 Output: $OUTPUT_DIR"
echo "========================================"

# 執行 run_ablation_b.py
python run_ablation_b.py \
    --output_dir "$OUTPUT_DIR" \
    --task "$TASK" \
    --subsets "$SUBSETS" \
    --limit "$LIMIT" \
    --eval_model "$EVAL_MODEL" \
    --opt_model "$OPT_MODEL" \
    --iterative \
    --iterative_count "$ITER_COUNT"