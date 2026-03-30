import os
import sys
import json

# 確保可以 import 專案內的模組
sys.path.append(os.getcwd())

# 引入現有的 BAKE 模組
from core.llm_client import LLMClient
from utils import config_loader
from regenerate_prompts import parse_generated_prompts

# ==========================================
# 參數設定區
# ==========================================
# DATASET_NAME = "high_school_microeconomics" # 用於替換 Prompt 中的變數，以及生成輸出檔名
DATASET_NAME = "miscellaneous" # 用於替換 Prompt 中的變數，以及生成輸出檔名
PROMPT_COUNT = 8

# 指定要讀取的 txt 模板檔名 (獨立出來，不再跟 DATASET_NAME 綁定)
INPUT_TEMPLATE_FILENAME = "generate.txt" 

# 指定實驗名稱
EXPERIMENT_NAME = "BAKE_qwen2.5-7b_qwen2.5-32b_MMLU_5Sub_Lim100_Base_0_Shuffle_20260306-130420"

# 路徑自動配置
BASE_EXPERIMENT_DIR = "/hcds_vol/private/emily/BAKE/experiments"
EXPERIMENT_DIR = os.path.join(BASE_EXPERIMENT_DIR, EXPERIMENT_NAME)

# 輸入檔案指向該實驗資料夾內部
RULE_FILE_PATH = os.path.join(EXPERIMENT_DIR, "final_rule.txt")
USER_PROMPT_FILE_PATH = os.path.join(EXPERIMENT_DIR, INPUT_TEMPLATE_FILENAME)
# ==========================================

def read_text_file(filepath: str) -> str:
    """讀取指定的 txt 檔案，若不存在則報錯並終止程式。"""
    if not os.path.exists(filepath):
        print(f"[錯誤] 找不到檔案: {filepath}")
        print(f"請確認該檔案確實存在於實驗資料夾中。")
        sys.exit(1)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    print("開始初始化設定...")
    print(f"目標實驗資料夾: {EXPERIMENT_DIR}")
    
    # 1. 讀取外部 txt 檔案
    print(f"正在讀取 Rule 檔案: {RULE_FILE_PATH}")
    rule_text = read_text_file(RULE_FILE_PATH)
    
    print(f"正在讀取 User Prompt 檔案: {USER_PROMPT_FILE_PATH}")
    user_prompt_template = read_text_file(USER_PROMPT_FILE_PATH)
    
    # 2. 載入 Config 與 Meta Prompts
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths'].get('meta_prompt_dir', 'meta_prompt'))
    
    # 3. 初始化您已有的 LLM Client (根據 config.yaml 設定)
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer'].get('pricing', {}))

    # 4. 準備 System Message (沿用 meta_prompts 中的設定)
    sys_tpl = meta_prompts.get("prompt_generation_system", "You are an expert Prompt Engineer.")
    try:
        sys_msg = sys_tpl.format(num=PROMPT_COUNT)
    except Exception:
        sys_msg = sys_tpl.replace("{num}", str(PROMPT_COUNT))

    # 5. 組合 User Message
    try:
        user_msg = user_prompt_template.format(
            dataset_name=DATASET_NAME,
            count=PROMPT_COUNT,
            rule_text=rule_text
        )
    except KeyError as e:
        print(f"[錯誤] User Prompt 格式錯誤！找不到變數: {e}")
        print("請確保 txt 中使用的變數標籤 (如 {dataset_name}) 是正確的。")
        sys.exit(1)

    # 6. 呼叫模型生成
    print(f"正在為 {DATASET_NAME} 呼叫模型生成 {PROMPT_COUNT} 個 Prompts...")
    raw_output = optimizer.chat(sys_msg, user_msg)

    # 7. 解析輸出
    prompts = parse_generated_prompts(raw_output)
    prompts = prompts[:PROMPT_COUNT]

    if not prompts:
        print("[錯誤] 無法解析出任何 Prompts！請檢查模型輸出格式。")
        print(f"模型原始輸出:\n{raw_output}")
        return

    # 8. 輸出存檔 (儲存至實驗資料夾)
    output_data = {
        "prompts": prompts
    }

    # 輸出檔名依然維持與 DATASET_NAME 綁定
    output_filename = f"{DATASET_NAME}_{EXPERIMENT_NAME}_prompts.json"
    output_filepath = os.path.join(EXPERIMENT_DIR, output_filename)
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"成功生成 {len(prompts)} 個 Prompts！")
    print(f"檔案已儲存至: {output_filepath}")

if __name__ == "__main__":
    main()