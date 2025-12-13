#!/bin/bash

# ==========================================
#  USER SETTINGS (使用者設定區)
# ==========================================

# 1. 設定要使用的 Ollama 模型名稱
MODEL_NAME="qwen2.5:7b"

# 2. 設定每個科目測試幾題 (設定小一點跑比較快，例如 10)
LIMIT=1

DATA_SPLIT="validation"

# 3. 設定要跑的資料集 (用空白隔開)
SUBJECTS=(
    "high_school_mathematics"
    "high_school_world_history"
    "high_school_physics"
    "professional_law"
    "business_ethics"
)

# 4. [新增] 設定要跑的實驗資料夾路徑 (請填入包含 optimized_prompts.txt 的資料夾)
#    你可以填入多個，腳本會依序執行
TARGET_FOLDERS=(
    "experiments/qwen2.5-7b_qwen2.5-32b_Limit100_IterOff_20251213-161414"
    "experiments/qwen2.5-7b_qwen2.5-32b_Limit100_IterOn_5_20251213-112715"
    # "experiments/你的新實驗資料夾..."
)

# 5. 結果輸出的資料夾名稱
OUTPUT_DIR="eval_results"

# ==========================================
#  SCRIPT LOGIC (以下無需修改)
# ==========================================

mkdir -p "$OUTPUT_DIR"

# 將陣列轉為空白分隔的字串
SUBJECTS_STRING="${SUBJECTS[*]}"

echo "Starting Batch Evaluation..."
echo "Model: $MODEL_NAME"
echo "Limit per subject: $LIMIT"
echo "Subjects: $SUBJECTS_STRING"
echo "Target Folders: ${#TARGET_FOLDERS[@]} folders defined."
echo "--------------------------------"

# 遍歷 TARGET_FOLDERS 陣列
for folder_path in "${TARGET_FOLDERS[@]}"
do
    # 移除路徑末端的 slash
    folder_path=${folder_path%/}
    
    if [ -d "$folder_path" ]; then
        echo "Processing folder: $folder_path"
        
        if [ -f "$folder_path/optimized_prompts.txt" ]; then
            python3 evaluate_prompts.py \
                --folder "$folder_path" \
                --model "$MODEL_NAME" \
                --limit "$LIMIT" \
                --output_dir "$OUTPUT_DIR" \
                --subjects $SUBJECTS_STRING
        else
            echo "[Warning] optimized_prompts.txt not found in $folder_path. Skipping."
        fi
        
    else
        echo "[Error] Directory not found: $folder_path"
    fi
    echo "--------------------------------"
done

echo "All evaluations finished."