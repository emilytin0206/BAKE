# main.py

import os # [新增]
from core.llm_client import LLMClient
from core.bake_engine import BakeEngine
from utils import config_loader, data_loader

def main():
    # 1. 初始化
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    
    scorer = LLMClient(cfg['scorer'], role='scorer', pricing=cfg['pricing']['scorer'])
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['pricing']['optimizer'])
    
    # 2. 載入資料
    dataset = data_loader.load_mixed_datasets(cfg['datasets'])
    
    # 3. 啟動引擎
    engine = BakeEngine(scorer, optimizer, cfg, meta_prompts)
    print(f"🚀 BAKE Engine Started with {len(dataset)} mixed samples...")
    
    # [修正] 接收兩個回傳值
    final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])
    
    # 4. 存檔 Prompts
    output_path = cfg['paths']['output_file']
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_prompts))
        
    # 5. [新增] 存檔 Final Rule
    # 我們將其存放在 logs 資料夾下，或是跟 output_file 同層級
    rule_path = "final_rule.txt" 
    with open(rule_path, "w", encoding="utf-8") as f:
        f.write(final_rule)
    
    # 6. 結算
    scorer.save_cost_record(cfg['paths']['cost_log'])
    optimizer.save_cost_record(cfg['paths']['cost_log'])
    
    print(f"\n✅ Prompts saved to: {output_path}")
    print(f"✅ Final Rule saved to: {rule_path}")
    print(f"💰 Scorer Cost: ${scorer.get_cost():.5f}")
    print(f"💰 Optimizer Cost: ${optimizer.get_cost():.5f}")

if __name__ == "__main__":
    main()