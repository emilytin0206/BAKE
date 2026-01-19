#!/bin/bash
set -e

echo "========================================"
echo "   BAKE Final Generation Debugger"
echo "========================================"

# 安裝依賴 (如果需要)
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi

# 執行 Debug Python Script
python3 debug_bake.py

echo "========================================"
echo "Done. Check 'final_prompt.txt' for results."
echo "========================================"