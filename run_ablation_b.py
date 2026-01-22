import os
import argparse
import sys
import yaml
import random
import json  # <--- [新增] 引入 json
from utils import config_loader, data_loader, logger
from core.llm_client import LLMClient
from core.bake_engine_ablation_b import ConciseBakeEngine

def main():
    parser = argparse.ArgumentParser(description='BAKE Ablation B: Concise Rule Experiment')
    
    # 參數定義
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task', type=str, default='mmlu')
    parser.add_argument('--subsets', type=str, default='all')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--opt_model', type=str, default=None)
    parser.add_argument('--iterative', action='store_true')
    parser.add_argument('--iterative_count', type=int, default=5)
    
    args = parser.parse_args()

    # 1. 載入基礎設定
    print(f"🔧 Loading Config: {args.config}")
    cfg = config_loader.load_config(args.config)
    
    # 2. 參數覆蓋 (CLI Override)
    if args.task:
        cfg['dataset']['active_task'] = args.task
        
    active_task = cfg['dataset'].get('active_task', 'mmlu')
    task_cfg = cfg['dataset'].get(active_task, {})

    # 設定 Subsets
    if args.subsets and active_task == 'mmlu':
        if args.subsets == 'all':
             task_cfg['subsets'] = ['all']
        else:
             task_cfg['subsets'] = [s.strip() for s in args.subsets.split(',')]

    # 設定 Limit, Split, Shuffle
    if args.limit is not None:
        task_cfg['limit'] = args.limit
    if args.split:
        task_cfg['split'] = args.split
    
    task_cfg['shuffle'] = args.shuffle

    # 寫回 Config
    cfg['dataset'][active_task] = task_cfg
    
    # 模型與迭代設定
    if args.eval_model: cfg['evaluation']['model_name'] = args.eval_model
    if args.opt_model: cfg['optimizer']['model_name'] = args.opt_model
    if args.iterative: cfg['bake']['iterative'] = True
    if args.iterative_count: cfg['bake']['iterative_prompt_count'] = args.iterative_count

    # 3. 建立目錄與路徑重導
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, "experiment_config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    if 'paths' not in cfg: cfg['paths'] = {}

    default_files = {
        'output_file': "optimized_prompts.txt",
        'detailed_log': "detailed_results.jsonl",
        'rules_log': "rules_history.txt",
        'cost_log': "cost_report.csv",
        'opt_status': "optimization_status.csv",
        'trace_log': "refinement_trace.jsonl",
        'prompt_history': "prompt_history.jsonl",
        'rule_evolution': "rule_evolution.jsonl"
    }
    for key, default_name in default_files.items():
        original_path = cfg['paths'].get(key, default_name)
        filename = os.path.basename(original_path)
        cfg['paths'][key] = os.path.join(args.output_dir, filename)

    # 4. 初始化
    scorer_cfg = cfg.get('evaluation', cfg.get('scorer'))
    scorer = LLMClient(scorer_cfg, role='scorer', pricing=scorer_cfg.get('pricing', {}))
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer']['pricing'])
    
    if hasattr(config_loader, 'load_meta_prompts'):
        meta_prompts = config_loader.load_meta_prompts(cfg['paths'].get('meta_prompt_dir', 'meta_prompt'))
    else:
        meta_prompts = {} 

    # 5. 載入資料
    print(f"📚 Loading Dataset [{active_task}]...")
    dataset = data_loader.load_specific_dataset(active_task, task_cfg)
    
    if not dataset:
        print("❌ Dataset empty. Exiting.")
        sys.exit(1)
    print(f"  📊 Final Dataset Size: {len(dataset)}")

    # 6. 執行 Engine
    print(f"🚀 Starting Ablation B (Concise Mode) | Output: {args.output_dir}")
    engine = ConciseBakeEngine(scorer, optimizer, cfg, meta_prompts)
    final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])

    # 7. 儲存結果
    
    # (A) 儲存 TXT (保持既有功能)
    with open(cfg['paths']['output_file'], "w", encoding="utf-8") as f:
        for p in final_prompts:
            f.write(p.replace('\n', '\\n') + "\n")

    # (B) 儲存 JSON (與 main.py 邏輯完全一致)
    # ==========================================
    exp_name = os.path.basename(os.path.normpath(args.output_dir))
    json_filename = f"{exp_name}.json"
    final_json_path = os.path.join(args.output_dir, json_filename)
    
    output_data = {
        "prompts": final_prompts
    }

    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"  💾 Final prompts saved to JSON: {final_json_path}")
    # ==========================================

    # 儲存 Rule 與 Cost
    with open(os.path.join(args.output_dir, "final_rule.txt"), "w", encoding="utf-8") as f:
        f.write(final_rule)
        
    scorer.save_cost_record(cfg['paths']['cost_log'])
    optimizer.save_cost_record(cfg['paths']['cost_log'])

    print("✅ Experiment Finished.")

if __name__ == "__main__":
    main()