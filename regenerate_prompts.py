import os
import sys
import argparse
import yaml
import re
import datetime

# 確保可以 import core 與 utils
sys.path.append(os.getcwd())

from core.llm_client import LLMClient
from utils import config_loader

def save_debug_log(file_path, title, content):
    """將除錯資訊寫入 Log 檔案"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*20} {title} [{timestamp}] {'='*20}\n")
        f.write(str(content))
        f.write(f"\n{'='*60}\n")

def parse_generated_prompts(raw_output):
    """
    從模型輸出中解析 Prompt (支援 <prompt> 標籤與傳統格式)
    """
    # 1. [優先] 嘗試使用 Regex 抓取 <prompt> 標籤內容
    # re.DOTALL 確保 . 可以匹配換行符號 (支援多行 Prompt)
    prompts = re.findall(r'<prompt>(.*?)</prompt>', raw_output, re.DOTALL | re.IGNORECASE)
    prompts = [p.strip() for p in prompts if p.strip()]

    # 2. [備案] 如果沒抓到標籤，使用舊的逐行解析邏輯 (Fallback)
    if not prompts:
        print("  [⚠️ Warning] No <prompt> tags found, using fallback line parsing.")
        for line in raw_output.split('\n'):
            line = line.strip()
            # 過濾太短或無意義的開頭
            if len(line) > 10 and not line.lower().startswith("here") and not line.lower().startswith("sure"):
                line = line.strip('"').strip("'")
                # 去除編號 (例如 "1. " 或 "1) ")
                if line[0].isdigit():
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        line = parts[-1].strip()
                    else:
                        parts = line.split(')', 1)
                        if len(parts) > 1:
                            line = parts[-1].strip()
                prompts.append(line)
                
    return prompts

def main():
    parser = argparse.ArgumentParser(description='BAKE Prompt Regenerator (In-Place Overwrite)')
    parser.add_argument('--rule_path', type=str, required=True, help='Path to the rule file')
    parser.add_argument('--count', type=int, default=8, help='Number of prompts to generate')
    
    args = parser.parse_args()

    # 1. 檢查輸入檔案
    if not os.path.exists(args.rule_path):
        print(f"❌ Error: Rule file not found at {args.rule_path}")
        sys.exit(1)

    target_dir = os.path.dirname(args.rule_path)
    output_path = os.path.join(target_dir, "optimized_prompts.txt")
    log_path = os.path.join(target_dir, "debug_generation_log.txt")  # <--- Log 檔案路徑

    print(f"📂 Target Directory: {target_dir}")
    print(f"📖 Reading Rule from: {os.path.basename(args.rule_path)}")
    print(f"📝 Debug Log will be saved to: {log_path}")

    # 清空舊的 log (可選，這裡選擇 append 模式方便查看多次嘗試，若要清空可 uncomment 下行)
    # open(log_path, 'w').close() 

    with open(args.rule_path, 'r', encoding='utf-8') as f:
        rule_text = f.read().strip()

    if not rule_text:
        print("❌ Error: Rule file is empty.")
        sys.exit(1)

    # 2. 載入 Config
    print("⚙️ Loading configuration...")
    try:
        cfg = config_loader.load_config()
        meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # 3. 初始化 LLM Client
    print(f"🤖 Initializing Optimizer Model ({cfg['optimizer']['model_name']})...")
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer']['pricing'])

    # 4. 準備 Prompt 模板
    sys_tpl = meta_prompts.get("prompt_generation_system", "You are a prompt engineer.")
    user_tpl = meta_prompts.get("prompt_generation_user", "Rule:\n{rule_text}\n\nGenerate {num} prompts.")

    try:
        # 注意：System Prompt 使用 {num}
        sys_msg = sys_tpl.format(num=args.count)
    except:
        sys_msg = sys_tpl.replace("{num}", str(args.count))

    try:
        # 注意：User Prompt 使用 {count}
        user_msg = user_tpl.format(rule_text=rule_text, count=args.count)
    except Exception:
        user_msg = f"Rule Guidelines:\n{rule_text}\n\nPlease generate {args.count} diverse prompts. Wrap each prompt in <prompt> tags."

    # ==========================================
    # 🔍 [DEBUG] 記錄 Prompt 到檔案
    # ==========================================
    print("  > Logging inputs to file...")
    save_debug_log(log_path, "INPUT: SYSTEM PROMPT", sys_msg)
    save_debug_log(log_path, "INPUT: USER PROMPT", user_msg)

    # 5. 執行生成
    print(f"🚀 Sending request to model (Generating {args.count} prompts)...")
    try:
        raw_output = optimizer.chat(sys_msg, user_msg)
    except Exception as e:
        print(f"❌ LLM Call Failed: {e}")
        sys.exit(1)

    # ==========================================
    # 🔍 [DEBUG] 記錄 Raw Output 到檔案
    # ==========================================
    print("  > Logging raw output to file...")
    save_debug_log(log_path, "OUTPUT: RAW MODEL RESPONSE", raw_output)

    # 6. 解析與覆蓋存檔
    prompts = parse_generated_prompts(raw_output)
    
    # 檢查解析結果
    if not prompts:
        print("❌ Warning: No prompts extracted! Check the debug log for raw output.")
    else:
        print(f"✅ Generated {len(prompts)} prompts.")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for p in prompts:
                f.write(p + "\n")
        print(f"💾 Overwritten successfully: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save output: {e}")

if __name__ == "__main__":
    main()