#!/bin/bash
# ==========================================
# 🧪 BAKE Base (不迭代) - 單一數學子集實驗
# 目的：證明單次行為對齊能以極低成本達到極高可解釋性
# ==========================================

TASK="mmlu"
SUBSETS="high_school_mathematics"  # 鎖定數學子集
SPLIT="train"                      # 在 train set 上進行錯誤診斷與規則提取
LIMIT=50                           # 給定 50 題作為分析基準即可 (成本極低)

ITERATIVE="false"                  # 🎯 關閉迭代 (Base Mode)
ITER_COUNT=0
SHUFFLE="true"

EVAL_MODEL="qwen2.5:7b"
OPT_MODEL="qwen2.5:32b"

DATE_LABEL=$(date +"%Y%m%d-%H%M")
EXP_NAME="BAKE_Base_Math_${DATE_LABEL}"
OUTPUT_DIR="experiments/${EXP_NAME}"

echo "========================================"
echo "🔥 Starting BAKE Base Experiment"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "🎯 Mode: Base (Zero-Shot Extraction)"
echo "========================================"

# 建構指令
CMD="python main.py \
    --output_dir $OUTPUT_DIR \
    --task $TASK \
    --subsets $SUBSETS \
    --split $SPLIT \
    --limit $LIMIT \
    --eval_model $EVAL_MODEL \
    --opt_model $OPT_MODEL \
    --shuffle"

echo "Running command: $CMD"
$CMD

# 實驗結束後提示
echo "----------------------------------------"
echo "✅ BAKE Base 提取完成！"
echo "📜 請查看 $OUTPUT_DIR/final_rule.txt 以獲取『可解釋性表格』所需的人類可讀規則。"
echo "💰 訓練成本紀錄於 $OUTPUT_DIR/cost_log.json"
echo "----------------------------------------"