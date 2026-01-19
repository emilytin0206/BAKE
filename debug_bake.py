import os
import json
import shutil
import yaml
import time
import sys
import re

# 確保引用路徑正確
sys.path.append(os.getcwd())

# 引用核心模組
from core.llm_client import LLMClient
from core.bake_engine import BakeEngine
from utils import config_loader, data_loader

# ==========================================
# 設定 Debug 輸出目錄
# ==========================================
DEBUG_ROOT = "debug_logs"

def setup_debug_dir():
    if os.path.exists(DEBUG_ROOT):
        shutil.rmtree(DEBUG_ROOT)
    os.makedirs(DEBUG_ROOT)
    print(f"📁 Debug logs will be saved to: {DEBUG_ROOT}/")

# ==========================================
# 攔截器 (Spy Logic) - 針對 LLMClient.chat
# ==========================================
original_chat = LLMClient.chat
step_counter = 0

def spied_chat(self, system_prompt: str, user_prompt: str):
    global step_counter
    step_counter += 1
    
    # 判斷當前步驟
    step_name = "unknown_step"
    if "evaluate" in system_prompt or "Helpful Assistant" in system_prompt:
        step_name = "evaluation"
    elif "DIAGNOSE" in system_prompt:
        step_name = "refinement"
    elif "underlying reasoning logic" in system_prompt:
        step_name = "rule_extraction"
    elif "integrate" in system_prompt:
        step_name = "rule_combination"
    elif "Visually and structurally different" in system_prompt:
        step_name = "prompt_generation_FINAL" # 標記這一步是生成 Prompt
    
    step_dir = os.path.join(DEBUG_ROOT, f"{step_counter:02d}_{step_name}")
    os.makedirs(step_dir, exist_ok=True)
    
    print(f"   --> [Step {step_counter}] {step_name} executing...")

    # 儲存 Input
    messages_snapshot = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    with open(os.path.join(step_dir, "input_messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages_snapshot, f, indent=2, ensure_ascii=False)
        
    # 執行原始 Chat
    try:
        start_time = time.time()
        response_text = original_chat(self, system_prompt, user_prompt)
        duration = time.time() - start_time
    except Exception as e:
        with open(os.path.join(step_dir, "ERROR.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))
        raise e

    # 儲存 Output
    with open(os.path.join(step_dir, "raw_output_from_llm.txt"), "w", encoding="utf-8") as f:
        f.write(response_text)

    # 儲存 Meta
    with open(os.path.join(step_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": getattr(self, "model_name", "unknown"),
            "role": getattr(self, "role", "unknown"),
            "duration_seconds": round(duration, 2)
        }, f, indent=2)

    return response_text

# ==========================================
# 主程式
# ==========================================
def main():
    setup_debug_dir()
    
    print("🔧 Injecting interceptor into LLMClient.chat...")
    LLMClient.chat = spied_chat

    # 1. 讀取 Config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("❌ Config file not found!")
        return

    print("⚙️ Loading Configuration...")
    cfg = config_loader.load_config(config_path)

    # ==========================================
    # [Debug 設定覆蓋]
    # ==========================================
    print("⚙️ Overriding config for debugging:")
    
    # 強制開啟迭代，確保會跑到 Final Generate
    cfg['bake']['iterative'] = True 
    cfg['bake']['iterative_prompt_count'] = 3
    
    # 只跑 1 題資料，加速流程
    cfg['dataset']['mmlu']['limit'] = 1 
    
    # 設定路徑
    if 'paths' not in cfg: cfg['paths'] = {}
    if 'meta_prompt_dir' not in cfg['paths']:
        cfg['paths']['meta_prompt_dir'] = 'meta_prompt'

    # 設定 Log 路徑 (確保我們知道檔案存在哪)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    cfg['paths']['opt_status'] = os.path.join(log_dir, "optimization_status.csv")
    
    # 2. 準備依賴物件
    print("📂 Loading Meta Prompts...")
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])

    print("🤖 Initializing LLM Clients...")
    scorer_cfg = cfg.get('evaluation', cfg.get('scorer'))
    scorer = LLMClient(scorer_cfg, role='scorer', pricing=scorer_cfg.get('pricing', {}))
    
    optimizer_cfg = cfg['optimizer']
    optimizer = LLMClient(optimizer_cfg, role='optimizer', pricing=optimizer_cfg.get('pricing', {}))

    print("📚 Loading Dataset...")
    active_task = cfg['dataset'].get('active_task', 'mmlu')
    task_cfg = cfg['dataset'].get(active_task, {})
    task_cfg['limit'] = 1  # 再次確保 dataset 只有一筆
    
    dataset = data_loader.load_specific_dataset(active_task, task_cfg)
    if not dataset:
        print("❌ No dataset loaded! Exiting.")
        return

    initial_prompts = cfg.get('initial_prompts', [])[:3]
    print(f"   - Initial Prompts ({len(initial_prompts)}): {initial_prompts}")

    # 3. 啟動 BAKE Engine
    print("\n🚀 Starting BAKE Engine (Debug Mode)...")
    try:
        # 正確傳入 4 個參數
        engine = BakeEngine(scorer, optimizer, cfg, meta_prompts)
        
        # 執行並接收回傳值 (final_prompts)
        final_prompts, final_rule = engine.run(dataset, initial_prompts)
        
        print("\n✅ Debug run completed!")
        
        # ==========================================
        # [驗證與轉存] 這裡處理您的 "final_prompt" 需求
        # ==========================================
        print("\n🔍 Verifying Final Generation Output...")
        
        if final_prompts and isinstance(final_prompts, list):
            print(f"  ✅ Success! Generated {len(final_prompts)} prompts (parsed as List).")
            print(f"  📝 Preview 1st prompt: {final_prompts[0][:60]}...")
            
            # 將結果轉存為 'final_prompt.txt'
            target_file = "final_prompt.txt"
            with open(target_file, "w", encoding="utf-8") as f:
                for p in final_prompts:
                    # 確保寫入時是壓平的一行
                    f.write(p.replace('\n', '\\n') + "\n")
            
            print(f"  💾 Saved final parsed prompts to: {os.path.abspath(target_file)}")
            
        else:
            print("  ⚠️ Warning: No prompts were returned. (Did the rule extraction fail?)")
            
        # 檢查原始 Engine 輸出的檔案是否存在
        original_output = os.path.join(log_dir, "final_optimized_prompts.txt")
        if os.path.exists(original_output):
            print(f"  (Original engine output found at: {original_output})")
            
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()