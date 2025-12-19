# main.py

import os
import argparse
import sys
import yaml  # [New] 務必加入這行
from core.llm_client import LLMClient
from core.bake_engine import BakeEngine
from utils import config_loader, data_loader

def parse_arguments():
    # ... (保持原樣)
    parser = argparse.ArgumentParser(description='BAKE Automation Runner')
    parser.add_argument('--scorer_model', type=str, help='Override scorer model name') # 如果您已經改名為 eval_model 請對應修改
    parser.add_argument('--eval_model', type=str, help='Override evaluation (scorer) model name') # 配合 BAKE.sh
    parser.add_argument('--optimizer_model', type=str, help='Override optimizer model name')
    parser.add_argument('--opt_model', type=str, help='Override optimizer model name') # 配合 BAKE.sh
    
    parser.add_argument('--dataset_limit', type=int, help='Override dataset limit per subset')
    parser.add_argument('--limit', type=int, help='Override dataset limit') # 配合 BAKE.sh
    
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save all outputs')
    parser.add_argument('--iterative', action='store_true', help='Enable iterative prompt updates based on rules')
    parser.add_argument('--iterative_prompt_count', type=int, help='Number of prompts to generate in iterative mode')
    parser.add_argument('--iterative_count', type=int, help='Number of prompts') # 配合 BAKE.sh
    
    # Dataset 相關
    parser.add_argument('--task', type=str, choices=['mmlu', 'gsm8k'], help='Choose active dataset')
    parser.add_argument('--subsets', type=str, help='Comma-separated subsets')
    parser.add_argument('--split', type=str, help='Override dataset split')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    
    # --- 1. 處理參數覆蓋 (CLI Override) ---
    # 支援新舊參數名稱，確保相容性
    eval_model = args.eval_model or args.scorer_model
    if eval_model:
        cfg['evaluation']['model_name'] = eval_model # 注意: 根據您的 config 結構可能是 cfg['scorer'] 或 cfg['evaluation']

    opt_model = args.opt_model or args.optimizer_model
    if opt_model:
        cfg['optimizer']['model_name'] = opt_model

    # Dataset 設定
    if args.task:
        cfg['dataset']['active_task'] = args.task
    
    active_task = cfg['dataset'].get('active_task', 'mmlu') # Default fallback
    task_cfg = cfg['dataset'].get(active_task, {}) # 取得該 task 的 dict

    limit = args.limit if args.limit is not None else args.dataset_limit
    if limit is not None:
        task_cfg['limit'] = limit
        
    if args.split:
        task_cfg['split'] = args.split
        
    if active_task == 'mmlu' and args.subsets:
        task_cfg['subsets'] = [s.strip() for s in args.subsets.split(',')]
        
    cfg['dataset'][active_task] = task_cfg # 寫回

    # 迭代設定
    cfg['bake']['iterative'] = args.iterative
    iter_count = args.iterative_count or args.iterative_prompt_count
    if iter_count:
        cfg['bake']['iterative_prompt_count'] = iter_count

    # --- 2. 目錄設定 ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ==========================================
    # [New] 儲存實驗當下的 Config 快照
    # ==========================================
    config_snapshot_path = os.path.join(args.output_dir, "experiment_config.yaml")
    print(f"💾 Saving experiment config to: {config_snapshot_path}")
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        # 使用 yaml.dump 將最終的 cfg 物件寫入檔案
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    # ==========================================
    
    # --- 3. 路徑重導 ---
    # 確保所有 log 都存到 output_dir
    for key in ['output_file', 'detailed_log', 'rules_log', 'cost_log', 'opt_status', 'trace_log', 'prompt_history', 'rule_evolution']:
        if key in cfg['paths']:
            filename = os.path.basename(cfg['paths'][key])
            cfg['paths'][key] = os.path.join(args.output_dir, filename)

    # --- 4. 初始化與執行 ---
    # 請根據您最新的 config 結構調整 key (例如 cfg['evaluation'] 或 cfg['scorer'])
    # 假設您已經更新為新結構：
    scorer_cfg = cfg.get('evaluation', cfg.get('scorer')) 
    scorer = LLMClient(scorer_cfg, role='scorer', pricing=scorer_cfg.get('pricing', {}))
    
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer']['pricing'])
    
    # 載入資料 (使用 data_loader 的新函式)
    dataset = data_loader.load_specific_dataset(active_task, task_cfg)
    
    engine = BakeEngine(scorer, optimizer, cfg, meta_prompts)
    print(f"🚀 BAKE Engine Started with {len(dataset)} samples...")
    
    try:
        final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])
        
        with open(cfg['paths']['output_file'], "w", encoding="utf-8") as f:
            f.write("\n".join(final_prompts))
            
        rule_path = os.path.join(args.output_dir, "final_rule.txt")
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(final_rule)
        
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