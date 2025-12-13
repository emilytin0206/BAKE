#!/bin/bash

# ==========================================
# BAKE 實驗自動化腳本 (v2: 含 Iterative 開關)
# ==========================================

# 1. 定義實驗參數陣列
# 格式：ScorerModel | OptimizerModel | Limit | EnableIterative(true/false)
EXPERIMENTS=(
    # 實驗 1: 關閉迭代，只跑流程 (Baseline)
    "qwen2.5:7b|qwen2.5:32b|100|true"
    
    # 實驗 2: 開啟迭代，測試熱替換效果
    "qwen2.5:7b|qwen2.5:32b|100|false"
)

# 基礎輸出目錄
BASE_DIR="experiments"
mkdir -p "$BASE_DIR"

echo "========================================"
echo "🚀 Starting Batch Experiments"
echo "Queue size: ${#EXPERIMENTS[@]}"
echo "========================================"

count=1
total=${#EXPERIMENTS[@]}

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r SCORER OPTIMIZER LIMIT ITERATIVE <<< "$exp"
    
    SAFE_SCORER=$(echo "$SCORER" | tr ':' '-')
    SAFE_OPT=$(echo "$OPTIMIZER" | tr ':' '-')
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    
    # 資料夾名稱加上模式標記 (IterOn/IterOff)
    if [ "$ITERATIVE" = "true" ]; then
        MODE_STR="IterOn"
        ITERATIVE_FLAG="--iterative"
    else
        MODE_STR="IterOff"
        ITERATIVE_FLAG=""
    fi
    
    DIR_NAME="${SAFE_SCORER}_${SAFE_OPT}_Limit${LIMIT}_${MODE_STR}_${TIMESTAMP}"
    OUTPUT_PATH="$BASE_DIR/$DIR_NAME"
    
    echo ""
    echo "[${count}/${total}] Running Experiment: $DIR_NAME"
    echo "   🔹 Scorer: $SCORER"
    echo "   🔹 Optimizer: $OPTIMIZER"
    echo "   🔹 Limit: $LIMIT"
    echo "   🔹 Mode: $MODE_STR"
    echo "   📂 Saving to: $OUTPUT_PATH"
    
    # 執行 Python (動態加入 --iterative)
    python3 main.py \
        --scorer_model "$SCORER" \
        --optimizer_model "$OPTIMIZER" \
        --dataset_limit "$LIMIT" \
        --output_dir "$OUTPUT_PATH" \
        $ITERATIVE_FLAG
        
    if [ $? -eq 0 ]; then
        echo "✅ Experiment ${count} Completed Successfully."
    else
        echo "❌ Experiment ${count} Failed."
    fi
    
    ((count++))
    sleep 2
done

echo ""
echo "🎉 All experiments finished!"