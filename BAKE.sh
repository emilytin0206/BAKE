#!/bin/bash

# ==========================================
# BAKE 實驗自動化腳本 (v3: 含 Iterative Count 標記)
# ==========================================

# 1. 定義實驗參數陣列
# 格式：Scorer | Optimizer | Limit | EnableIterative | IterCount(新參數)
EXPERIMENTS=(
    
    "qwen2.5:7b|qwen2.5:32b|300|true|5"
    "qwen2.5:7b|qwen2.5:32b|300|false|5"

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
    # [修改] 讀取第 5 個參數 ITER_COUNT
    IFS='|' read -r SCORER OPTIMIZER LIMIT ITERATIVE ITER_COUNT <<< "$exp"
    
    SAFE_SCORER=$(echo "$SCORER" | tr ':' '-')
    SAFE_OPT=$(echo "$OPTIMIZER" | tr ':' '-')
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    
    # [修改] 檔名與參數邏輯
    if [ "$ITERATIVE" = "true" ]; then
        # 檔名加上數量，例如: IterOn_5
        MODE_STR="IterOn_${ITER_COUNT}"
        # 傳遞參數給 main.py
        ITERATIVE_FLAG="--iterative --iterative_prompt_count $ITER_COUNT"
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
    echo "   🔹 Mode: $MODE_STR (Count: $ITER_COUNT)"
    echo "   📂 Saving to: $OUTPUT_PATH"
    
    # 執行 Python
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