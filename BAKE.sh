#!/bin/bash

# ==========================================
# BAKE 實驗自動化腳本 (含時間戳記)
# ==========================================

# 1. 定義實驗參數陣列
# 格式：ScorerModel | OptimizerModel | Limit
EXPERIMENTS=(
    "qwen2.5:7b|qwen2.5:32b|1"
)

# 基礎輸出目錄
BASE_DIR="experiments"

# 建立基礎目錄
mkdir -p "$BASE_DIR"

echo "========================================"
echo "🚀 Starting Batch Experiments"
echo "Queue size: ${#EXPERIMENTS[@]}"
echo "========================================"

# 2. 迴圈執行實驗
count=1
total=${#EXPERIMENTS[@]}

for exp in "${EXPERIMENTS[@]}"; do
    # 解析參數 (使用 | 分隔)
    IFS='|' read -r SCORER OPTIMIZER LIMIT <<< "$exp"
    
    # 處理檔名 (將 : 替換為 - 以避免路徑錯誤)
    SAFE_SCORER=$(echo "$SCORER" | tr ':' '-')
    SAFE_OPT=$(echo "$OPTIMIZER" | tr ':' '-')
    
    # [新增] 取得當前時間戳記 (例如: 20251213-103000)
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    
    # [修改] 自動產生資料夾名稱，加上時間戳記
    # 格式: <Scorer>_<Optimizer>_Limit<N>_<Time>
    DIR_NAME="${SAFE_SCORER}_${SAFE_OPT}_Limit${LIMIT}_${TIMESTAMP}"
    OUTPUT_PATH="$BASE_DIR/$DIR_NAME"
    
    echo ""
    echo "[${count}/${total}] Running Experiment: $DIR_NAME"
    echo "   🔹 Scorer: $SCORER"
    echo "   🔹 Optimizer: $OPTIMIZER"
    echo "   🔹 Limit: $LIMIT"
    echo "   📂 Saving to: $OUTPUT_PATH"
    
    # 3. 呼叫 Python 腳本
    # 注意：請確保您的 main.py 已經更新為支援 argparse 的版本
    python3 main.py \
        --scorer_model "$SCORER" \
        --optimizer_model "$OPTIMIZER" \
        --dataset_limit "$LIMIT" \
        --output_dir "$OUTPUT_PATH"
        
    # 檢查執行結果
    if [ $? -eq 0 ]; then
        echo "✅ Experiment ${count} Completed Successfully."
    else
        echo "❌ Experiment ${count} Failed."
    fi
    
    ((count++))
    sleep 2 # 休息一下
done

echo ""
echo "🎉 All experiments finished!"