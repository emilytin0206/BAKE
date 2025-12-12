from core.llm_client import LLMClient
from core.bake_engine import BakeEngine
from utils import config_loader, data_loader

def main():
    # 1. 初始化
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    
    scorer = LLMClient(cfg['scorer'], role='scorer', pricing=cfg['pricing']['scorer'])
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['pricing']['optimizer'])
    
    # 2. [關鍵修改] 載入混合資料集
    # 現在 dataset 是一個 list，包含數學與選擇題
    dataset = data_loader.load_mixed_datasets(cfg['datasets'])
    
    # 3. 啟動引擎
    engine = BakeEngine(scorer, optimizer, cfg, meta_prompts)
    print(f"🚀 BAKE Engine Started with {len(dataset)} mixed samples...")
    
    final_prompts = engine.run(dataset, cfg['initial_prompts'])
    
    # 5. 存檔與結算
    output_path = cfg['paths']['output_file']
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_prompts))
    
    # 6. 自動記錄成本 (CSV)
    scorer.save_cost_record(cfg['paths']['cost_log'])
    optimizer.save_cost_record(cfg['paths']['cost_log'])
    
    print(f"\n✅ Done! Prompts saved to {output_path}")
    print(f"💰 Cost Log saved to {cfg['paths']['cost_log']}")
    print(f"   - Scorer Cost: ${scorer.get_cost():.5f}")
    print(f"   - Optimizer Cost: ${optimizer.get_cost():.5f}")

if __name__ == "__main__":
    main()