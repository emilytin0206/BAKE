#!/bin/bash
# ==========================================
# 🧪 BAKE Experiment Runner (Single Run)
# ==========================================

# 1. 參數設定 (Settings)
# ------------------------------------------
TASK="mmlu"
SUBSETS="high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics"
SPLIT="test"
LIMIT=100
ITERATIVE="true"
ITER_COUNT=5
SHUFFLE="true"
EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"

# 指定單一來源 (Single Source)
INIT_SRC="gpt4o"

echo "############################################################"
echo "▶️  Starting Single Experiment for Source: $INIT_SRC"
echo "############################################################"

# 2. 更新 Config (呼叫 update_config.py)
# ------------------------------------------
echo "🔄 Updating config.yaml with prompts from: $INIT_SRC"
python3 update_config.py "$INIT_SRC"

if [ $? -ne 0 ]; then
    echo "❌ Failed to update config for $INIT_SRC. Exiting..."
    exit 1
fi

# 3. 自動命名邏輯 (Auto-Naming)
# ------------------------------------------
T_MODEL_SAFE=${EVAL_MODEL//:/-}
O_MODEL_SAFE=${OPT_MODEL//:/-}

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

LIM_LABEL="Lim${LIMIT}"

if [ "$ITERATIVE" == "true" ]; then
    MODE_LABEL="Iter${ITER_COUNT}"
else
    MODE_LABEL="Base"
fi

if [ "$SHUFFLE" == "true" ]; then
    SHUFFLE_LABEL="Shuffle"
else
    SHUFFLE_LABEL="Seq"
fi

# 來源標籤 (手動處理首字母大寫)
if [ "$INIT_SRC" == "gpt4o" ]; then 
    INIT_LABEL="InitGpt4o"
else 
    INIT_LABEL="InitGemini"
fi

DATE_LABEL=$(date +"%Y%m%d-%H%M%S")

# 組合最終實驗資料夾名稱
EXP_NAME="BAKE_${INIT_LABEL}_${T_MODEL_SAFE}_${O_MODEL_SAFE}_${DS_LABEL}_${SUB_LABEL}_${LIM_LABEL}_${MODE_LABEL}_${SHUFFLE_LABEL}_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

# 4. 建構與執行指令
# ------------------------------------------
echo "----------------------------------------"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "----------------------------------------"

CMD="python main.py --output_dir $OUTPUT_DIR --task $TASK --limit $LIMIT --split $SPLIT"

if [ "$TASK" == "mmlu" ]; then CMD="$CMD --subsets $SUBSETS"; fi
if [ ! -z "$EVAL_MODEL" ]; then CMD="$CMD --eval_model $EVAL_MODEL"; fi
if [ ! -z "$OPT_MODEL" ]; then CMD="$CMD --opt_model $OPT_MODEL"; fi
if [ "$ITERATIVE" == "true" ]; then CMD="$CMD --iterative --iterative_count $ITER_COUNT"; fi
if [ "$SHUFFLE" == "true" ]; then CMD="$CMD --shuffle"; fi

echo "Running: $CMD"
$CMD

if [ $? -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
else
    echo "❌ Experiment failed."
    exit 1
fi



# #!/bin/bash
# # ==========================================
# # 🧪 BAKE Experiment Runner (Run ALL Sources)
# # ==========================================

# # 1. 參數設定 (Settings)
# # ------------------------------------------
# TASK="mmlu"
# SUBSETS="high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics"
# SPLIT="test"
# LIMIT=100
# ITERATIVE="true"
# ITER_COUNT=5
# SHUFFLE="true"
# EVAL_MODEL="qwen2.5:7b"
# OPT_MODEL="qwen2.5:32b"

# # 定義要跑的 Initial Prompt 來源列表
# SOURCES=("gpt4o" "gemini")
# # SOURCES=("gpt4o")

# # ==========================================
# # 🚀 Loop Through Sources
# # ==========================================

# for INIT_SRC in "${SOURCES[@]}"; do
#     echo "############################################################"
#     echo "▶️  Starting Sequence for Source: $INIT_SRC"
#     echo "############################################################"

#     # 1. 更新 Config (呼叫 update_config.py)
#     echo "🔄 Updating config.yaml with prompts from: $INIT_SRC"
#     python3 update_config.py "$INIT_SRC"

#     if [ $? -ne 0 ]; then
#         echo "❌ Failed to update config for $INIT_SRC. Skipping..."
#         continue
#     fi

#     # 2. 自動命名邏輯 (Auto-Naming)
#     T_MODEL_SAFE=${EVAL_MODEL//:/-}
#     O_MODEL_SAFE=${OPT_MODEL//:/-}

#     if [ "$TASK" == "mmlu" ]; then
#         DS_LABEL="MMLU"
#         if [ "$SUBSETS" == "all" ]; then
#             SUB_LABEL="All"
#         else
#             IFS=',' read -ra ADDR <<< "$SUBSETS"
#             COUNT=${#ADDR[@]}
#             SUB_LABEL="${COUNT}Sub"
#         fi
#     else
#         DS_LABEL="${TASK^^}"
#         SUB_LABEL="NA"
#     fi

#     LIM_LABEL="Lim${LIMIT}"

#     if [ "$ITERATIVE" == "true" ]; then
#         MODE_LABEL="Iter${ITER_COUNT}"
#     else
#         MODE_LABEL="Base"
#     fi

#     if [ "$SHUFFLE" == "true" ]; then
#         SHUFFLE_LABEL="Shuffle"
#     else
#         SHUFFLE_LABEL="Seq"
#     fi

#     # 來源標籤 (手動處理首字母大寫)
#     if [ "$INIT_SRC" == "gpt4o" ]; then 
#         INIT_LABEL="InitGpt4o"
#     else 
#         INIT_LABEL="InitGemini"
#     fi

#     DATE_LABEL=$(date +"%Y%m%d-%H%M%S")
    
#     # 組合最終實驗資料夾名稱
#     EXP_NAME="BAKE_${INIT_LABEL}_${T_MODEL_SAFE}_${O_MODEL_SAFE}_${DS_LABEL}_${SUB_LABEL}_${LIM_LABEL}_${MODE_LABEL}_${SHUFFLE_LABEL}_${DATE_LABEL}"
#     OUTPUT_DIR="experiments/${EXP_NAME}"

#     # 3. 建構與執行指令
#     echo "----------------------------------------"
#     echo "📂 Output Dir: $OUTPUT_DIR"
#     echo "----------------------------------------"

#     CMD="python main.py --output_dir $OUTPUT_DIR --task $TASK --limit $LIMIT --split $SPLIT"

#     if [ "$TASK" == "mmlu" ]; then CMD="$CMD --subsets $SUBSETS"; fi
#     if [ ! -z "$EVAL_MODEL" ]; then CMD="$CMD --eval_model $EVAL_MODEL"; fi
#     if [ ! -z "$OPT_MODEL" ]; then CMD="$CMD --opt_model $OPT_MODEL"; fi
#     if [ "$ITERATIVE" == "true" ]; then CMD="$CMD --iterative --iterative_count $ITER_COUNT"; fi
#     if [ "$SHUFFLE" == "true" ]; then CMD="$CMD --shuffle"; fi

#     echo "Running: $CMD"
#     $CMD

#     if [ $? -eq 0 ]; then
#         echo "✅ Experiment for $INIT_SRC completed successfully!"
#     else
#         echo "❌ Experiment for $INIT_SRC failed."
#     fi
    
#     echo ""
#     echo "⏳ Waiting 5 seconds before next run..."
#     sleep 5
# done

# echo "🎉 All experiments finished!"