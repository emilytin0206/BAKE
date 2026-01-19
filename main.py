# main.py

import os
import argparse
import sys
import yaml
import copy
import json  # <--- [新增] 引入 json 模組
from core.llm_client import LLMClient
from core.bake_engine import BakeEngine
from utils import config_loader, data_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description='BAKE Automation Runner')
    parser.add_argument('--scorer_model', type=str, help='Override scorer model name') 
    parser.add_argument('--eval_model', type=str, help='Override evaluation (scorer) model name') 
    parser.add_argument('--optimizer_model', type=str, help='Override optimizer model name')
    parser.add_argument('--opt_model', type=str, help='Override optimizer model name') 
    
    parser.add_argument('--dataset_limit', type=int, help='Override dataset limit per subset')
    parser.add_argument('--limit', type=int, help='Override dataset limit') 
    
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save all outputs')
    parser.add_argument('--iterative', action='store_true', help='Enable iterative prompt updates based on rules')
    parser.add_argument('--iterative_prompt_count', type=int, help='Number of prompts to generate in iterative mode')
    parser.add_argument('--iterative_count', type=int, help='Number of prompts') 
    
    # Dataset 相關
    parser.add_argument('--task', type=str, choices=['mmlu', 'gsm8k'], help='Choose active dataset')
    parser.add_argument('--subsets', type=str, help='Comma-separated subsets')
    parser.add_argument('--split', type=str, help='Override dataset split')

    # 模式參數: 僅保留 shuffle
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset (Mix all samples)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 載入基礎設定
    cfg = config_loader.load_config()
    meta_prompts = config_loader.load_meta_prompts(cfg['paths']['meta_prompt_dir'])
    
    # --- 1. 處理參數覆蓋 (CLI Override) ---
    eval_model = args.eval_model or args.scorer_model
    if eval_model:
        cfg['evaluation']['model_name'] = eval_model 

    opt_model = args.opt_model or args.optimizer_model
    if opt_model:
        cfg['optimizer']['model_name'] = opt_model

    # Dataset 設定
    if args.task:
        cfg['dataset']['active_task'] = args.task
    
    active_task = cfg['dataset'].get('active_task', 'mmlu') 
    task_cfg = cfg['dataset'].get(active_task, {}) 

    limit = args.limit if args.limit is not None else args.dataset_limit
    if limit is not None:
        task_cfg['limit'] = limit
        
    if args.split:
        task_cfg['split'] = args.split
        
    if active_task == 'mmlu' and args.subsets:
        task_cfg['subsets'] = [s.strip() for s in args.subsets.split(',')]
    
    # 設定 Shuffle (由 CLI 參數覆蓋 config)
    task_cfg['shuffle'] = args.shuffle

    # 寫回主設定
    cfg['dataset'][active_task] = task_cfg 

    # 迭代設定
    cfg['bake']['iterative'] = args.iterative
    iter_count = args.iterative_count or args.iterative_prompt_count
    if iter_count:
        cfg['bake']['iterative_prompt_count'] = iter_count

    # --- 2. 準備輸出目錄 ---
    print(f"\n{'='*50}")
    print(f"📂 Starting Experiment in: {args.output_dir}")
    print(f"{'='*50}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 儲存 Config 快照
    config_snapshot_path = os.path.join(args.output_dir, "experiment_config.yaml")
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    # --- 3. 路徑重導 ---
    # 將 log 路徑導向 output_dir
    # 注意：雖然這裡有處理 output_file，但最後我們會用新的 JSON 邏輯覆蓋它
    for key in ['output_file', 'detailed_log', 'rules_log', 'cost_log', 'opt_status', 'trace_log', 'prompt_history', 'rule_evolution']:
        if key in cfg['paths']:
            filename = os.path.basename(cfg['paths'][key])
            cfg['paths'][key] = os.path.join(args.output_dir, filename)

    # --- 4. 載入資料 ---
    dataset = data_loader.load_specific_dataset(active_task, task_cfg)
    
    if not dataset:
        print("❌ No data loaded. Exiting.")
        sys.exit(1)

    # --- 5. 初始化與執行 ---
    scorer_cfg = cfg.get('evaluation', cfg.get('scorer')) 
    scorer = LLMClient(scorer_cfg, role='scorer', pricing=scorer_cfg.get('pricing', {}))
    
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer']['pricing'])
    
    engine = BakeEngine(scorer, optimizer, cfg, meta_prompts)
    
    try:
        final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])
        
        # ==========================================
        # [修改] 儲存結果為 JSON 格式
        # ==========================================
        # 取得實驗名稱 (通常是 output_dir 的最後一層目錄名)
        exp_name = os.path.basename(os.path.normpath(args.output_dir))
        
        # 構建 JSON 檔名: 實驗名稱.json (例如: 20260116_test.json)
        json_filename = f"{exp_name}.json"
        final_json_path = os.path.join(args.output_dir, json_filename)
        
        output_data = {
            "prompts": final_prompts
        }

        # 使用 json.dump 輸出
        # indent=4: 縮排整齊
        # ensure_ascii=False: 支援非 ASCII (如中文)
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"  💾 Final prompts saved to JSON: {final_json_path}")
        
        # 同時保留儲存 Rule 內容
        rule_path = os.path.join(args.output_dir, "final_rule.txt")
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(final_rule)
        
        scorer.save_cost_record(cfg['paths']['cost_log'])
        optimizer.save_cost_record(cfg['paths']['cost_log'])
        
        print(f"✅ Experiment Success: {args.output_dir}")

    except Exception as e:
        print(f"❌ Experiment Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()