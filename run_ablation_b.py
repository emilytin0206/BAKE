import os
import argparse
import sys
import random  # <--- [新增] 用於打散資料
from utils import config_loader, data_loader, logger
from core.llm_client import LLMClient
from core.bake_engine_ablation_b import ConciseBakeEngine

def main():
    parser = argparse.ArgumentParser(description='BAKE Ablation B: Concise Rule Experiment')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task', type=str, default='mmlu')
    parser.add_argument('--subsets', type=str, default='all')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--split', type=str, default='test')      # <--- [新增]
    parser.add_argument('--shuffle', action='store_true')         # <--- [新增]
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--opt_model', type=str, default=None)
    parser.add_argument('--iterative', action='store_true')
    parser.add_argument('--iterative_count', type=int, default=5)
    
    args = parser.parse_args()

    # 1. Load Config
    cfg = config_loader.load_config()
    
    # Override Config
    if args.eval_model: cfg['evaluation']['model_name'] = args.eval_model
    if args.opt_model: cfg['optimizer']['model_name'] = args.opt_model
    cfg['bake']['iterative'] = args.iterative
    cfg['bake']['iterative_prompt_count'] = args.iterative_count
    
    # Update Dataset Config
    cfg['dataset'][args.task]['split'] = args.split
    cfg['dataset'][args.task]['shuffle'] = args.shuffle

    # Setup Paths
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for key in ['detailed_log', 'rules_log', 'opt_status', 'trace_log', 'prompt_history', 'rule_evolution']:
        filename = os.path.basename(cfg['paths'][key])
        cfg['paths'][key] = os.path.join(args.output_dir, filename)

    # 2. Init Models
    scorer = LLMClient(cfg['evaluation'], role='scorer')
    optimizer = LLMClient(cfg['optimizer'], role='optimizer')
    
    # 3. Load Meta Prompts
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    
    # 4. Load Data
    subset_list = args.subsets.split(',') if args.subsets != 'all' else 'all'
    
    # 載入資料 (這裡假設 data_loader 不一定有 shuffle 參數，我們手動處理最保險)
    dataset = data_loader.load_dataset_by_task(args.task, subset_list, args.split, args.limit)
    
    if args.shuffle:
        print("  🔀 Shuffling dataset...")
        random.seed(42) # 固定種子以確保可重現性
        random.shuffle(dataset)

    if not dataset:
        print("❌ Dataset empty.")
        sys.exit(1)

    print(f"  📊 Dataset Size: {len(dataset)} | Split: {args.split} | Shuffle: {args.shuffle}")

    # 5. Run Engine (Concise Mode)
    print(f"🚀 Starting Ablation B (Concise Mode) | Output: {args.output_dir}")
    
    engine = ConciseBakeEngine(scorer, optimizer, cfg, meta_prompts)
    final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])

    # Save Final Results
    with open(os.path.join(args.output_dir, "optimized_prompts.txt"), "w", encoding="utf-8") as f:
        for p in final_prompts:
            # 確保壓平 (Flatten) 後存檔
            f.write(p.replace('\n', '\\n') + "\n")

    with open(os.path.join(args.output_dir, "final_rule.txt"), "w", encoding="utf-8") as f:
        f.write(final_rule)

    print("✅ Experiment Finished.")

if __name__ == "__main__":
    main()