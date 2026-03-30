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
# ⚙️ 參數設定區
# ==========================================
# 1. 您的變數參數 (這裡已經改成 dataset_name，完美對應 txt 模板)
PARAMS = {
    "dataset_name": "high_school_microeconomics",  
    "count": 8                        
}

# 2. 檔案路徑設定
TEMPLATE_FILE_PATH = "pd.txt"

# 輸出：生成的 JSON 存檔名稱
OUTPUT_FILE_NAME = f"pd_{PARAMS['dataset_name']}.json"
# ==========================================

def read_text_file(filepath: str) -> str:
    """讀取指定的 txt 檔案，若不存在則報錯並終止程式。"""
    if not os.path.exists(filepath):
        print(f"[錯誤] 找不到模板檔案: {filepath}")
        print("請確認檔案是否存在。")
        sys.exit(1)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    print("🚀 開始初始化...")
    
    # 1. 讀取模板檔案
    print(f"📄 正在讀取模板: {TEMPLATE_FILE_PATH}")
    template_text = read_text_file(TEMPLATE_FILE_PATH)
    
    # 2. 載入 Config 與 Meta Prompts
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths'].get('meta_prompt_dir', 'meta_prompt'))
    
    # 3. 初始化 Optimizer (LLM Client)
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer'].get('pricing', {}))

    # 4. 準備 System Message
    sys_tpl = meta_prompts.get("prompt_generation_system", "You are an expert Prompt Engineer.")
    try:
        sys_msg = sys_tpl.format(num=PARAMS['count'])
    except Exception:
        sys_msg = sys_tpl.replace("{num}", str(PARAMS['count']))

    # 5. 組合 User Message
    try:
        # 這裡會自動把 PARAMS 裡的 "dataset_name" 填入 txt 的 {dataset_name} 中
        user_msg = template_text.format(**PARAMS)
    except KeyError as e:
        print(f"[錯誤] 模板格式錯誤！txt 檔案中包含了未定義的變數: {e}")
        print(f"目前可用的變數有: {list(PARAMS.keys())}")
        sys.exit(1)

    # 6. 呼叫模型生成
    dataset_val = PARAMS.get('dataset_name', 'unknown')
    count_val = PARAMS.get('count', 0)
    print(f"🧠 正在要求模型生成 {count_val} 個關於 {dataset_val} 的 Prompts...")
    
    raw_output = optimizer.chat(sys_msg, user_msg)

    # 7. 解析輸出
    prompts = parse_generated_prompts(raw_output)
    prompts = prompts[:count_val]

    if not prompts:
        print("[錯誤] 無法解析出任何 Prompts！請檢查模型輸出格式。")
        print(f"模型原始輸出:\n{raw_output}")
        return

    # 8. 輸出存檔
    output_data = {
        "prompts": prompts
    }

    with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ 成功生成 {len(prompts)} 個 Prompts！")
    print(f"💾 檔案已儲存至目前目錄: {OUTPUT_FILE_NAME}")

if __name__ == "__main__":
    main()