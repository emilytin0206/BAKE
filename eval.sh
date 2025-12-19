#!/bin/bash

# ==========================================
#  USER SETTINGS (使用者設定區)
# ==========================================

# 1. [關鍵修正] 設定遠端 Ollama API URL (原生 API 路徑)
#    對應你的 python code: http://140.113.86.14:11434/api/chat
API_URL="http://140.113.86.14:11434/api/chat"

# 2. [關鍵修正] 模型名稱 (使用伺服器上存在的模型，如 qwen2.5:14b)
MODEL_NAME="qwen2.5:7b"

# 3. 資料集設定 (Train on Test, Test on Validation)
DATA_SPLIT="validation"
LIMIT=300

# 4. 設定要評測的科目
SUBJECTS=(
    "high_school_mathematics"
    "high_school_world_history"
    "high_school_physics"
    "professional_law"
    "business_ethics"
)

# 5. 設定要跑的實驗資料夾
TARGET_FOLDERS=(
    "experiments/qwen2.5-7b_qwen2.5-32b_Limit100_IterOff_20251213-161414"
    "experiments/qwen2.5-7b_qwen2.5-32b_Limit100_IterOn_5_20251213-112715"

)

OUTPUT_DIR="eval_results"

# ==========================================
#  SCRIPT LOGIC
# ==========================================

mkdir -p "$OUTPUT_DIR"
SUBJECTS_STRING="${SUBJECTS[*]}"

echo "Starting Batch Evaluation..."
echo "Interface: $API_URL"
echo "Model: $MODEL_NAME"
echo "Split: $DATA_SPLIT"
echo "--------------------------------"

for folder_path in "${TARGET_FOLDERS[@]}"
do
    folder_path=${folder_path%/}
    
    if [ -d "$folder_path" ]; then
        echo "Processing: $folder_path"
        
        if [ -f "$folder_path/optimized_prompts.txt" ]; then
            python3 evaluate_prompts.py \
                --folder "$folder_path" \
                --model "$MODEL_NAME" \
                --api_url "$API_URL" \
                --limit "$LIMIT" \
                --split "$DATA_SPLIT" \
                --output_dir "$OUTPUT_DIR" \
                --subjects $SUBJECTS_STRING
        else
            echo "[Skip] No optimized_prompts.txt"
        fi
        
    else
        echo "[Error] Directory not found: $folder_path"
    fi
    echo "--------------------------------"
done

echo "Done."