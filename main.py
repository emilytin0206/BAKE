# main.py

import os
import argparse
import sys
from core.llm_client import LLMClient
from core.bake_engine import BakeEngine
from utils import config_loader, data_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description='BAKE Automation Runner')
    
    parser.add_argument('--scorer_model', type=str, help='Override scorer model name')
    parser.add_argument('--optimizer_model', type=str, help='Override optimizer model name')
    parser.add_argument('--dataset_limit', type=int, help='Override dataset limit per subset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save all outputs')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    
    # 覆蓋參數
    if args.scorer_model:
        cfg['scorer']['model_name'] = args.scorer_model
    if args.optimizer_model:
        cfg['optimizer']['model_name'] = args.optimizer_model
    if args.dataset_limit:
        for ds in cfg['datasets']:
            ds['limit'] = args.dataset_limit

    # 建立目錄
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"📂 Output Directory: {args.output_dir}")

    # 設定輸出路徑
    cfg['paths']['output_file'] = os.path.join(args.output_dir, "optimized_prompts.txt")
    cfg['paths']['detailed_log'] = os.path.join(args.output_dir, "detailed_results.jsonl")
    cfg['paths']['rules_log'] = os.path.join(args.output_dir, "rules_history.txt")
    cfg['paths']['cost_log'] = os.path.join(args.output_dir, "cost_report.csv")
    cfg['paths']['opt_status'] = os.path.join(args.output_dir, "optimization_status.csv")
    cfg['paths']['trace_log'] = os.path.join(args.output_dir, "refinement_trace.jsonl")
    
    # [新增] 兩個新的 Log 路徑
    cfg['paths']['prompt_history'] = os.path.join(args.output_dir, "prompt_history.jsonl")
    cfg['paths']['rule_evolution'] = os.path.join(args.output_dir, "rule_evolution.jsonl")

    # 初始化
    scorer = LLMClient(cfg['scorer'], role='scorer', pricing=cfg['pricing']['scorer'])
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['pricing']['optimizer'])
    
    dataset = data_loader.load_mixed_datasets(cfg['datasets'])
    
    engine = BakeEngine(scorer, optimizer, cfg, meta_prompts)
    print(f"🚀 BAKE Engine Started with {len(dataset)} samples...")
    
    try:
        final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])
        
        # 存檔 Prompts
        with open(cfg['paths']['output_file'], "w", encoding="utf-8") as f:
            f.write("\n".join(final_prompts))
            
        # 存檔 Final Rule
        rule_path = os.path.join(args.output_dir, "final_rule.txt")
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(final_rule)
        
        # 結算
        scorer.save_cost_record(cfg['paths']['cost_log'])
        optimizer.save_cost_record(cfg['paths']['cost_log'])
        
        print(f"\n✅ Experiment Success!")
        print(f"   Saved to: {args.output_dir}")

    except Exception as e:
        print(f"\n❌ Experiment Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()