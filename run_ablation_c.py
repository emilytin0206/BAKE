import argparse
import yaml
import os
import sys
import json # <--- [新增] 引入 json

from utils import config_loader, data_loader, logger
from core.llm_client import LLMClient
from core.bake_engine_ablation_c import SuccessOnlyBakeEngine 

def main():
    parser = argparse.ArgumentParser(description="BAKE Ablation Study Runner (Success-Only)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    
    parser.add_argument("--task", type=str, help="Override task in config")
    parser.add_argument("--subsets", type=str, help="Override subsets (comma separated)")
    parser.add_argument("--limit", type=int, help="Override dataset limit")
    parser.add_argument("--split", type=str, help="Override dataset split")
    parser.add_argument("--eval_model", type=str, help="Override evaluation model")
    parser.add_argument("--opt_model", type=str, help="Override optimizer model")
    parser.add_argument("--iterative", action='store_true', help="Enable iterative mode")
    parser.add_argument("--iterative_count", type=int, help="Number of prompts to generate in iterative mode")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the dataset (Mix all samples)")

    args = parser.parse_args()

    # 1. 載入設定
    print(f"🔧 Loading Config: {args.config}")
    cfg = config_loader.load_config(args.config)
    
    # 2. 處理參數覆蓋
    if args.task:
        cfg['dataset']['active_task'] = args.task
    
    active_task = cfg['dataset'].get('active_task', 'mmlu')
    task_cfg = cfg['dataset'].get(active_task, {})

    if args.subsets and active_task == 'mmlu':
        task_cfg['subsets'] = [s.strip() for s in args.subsets.split(',')]
    if args.limit is not None:
        task_cfg['limit'] = args.limit
    if args.split:
        task_cfg['split'] = args.split
    
    task_cfg['shuffle'] = args.shuffle
    cfg['dataset'][active_task] = task_cfg 

    if args.eval_model:
        if 'evaluation' in cfg:
            cfg['evaluation']['model_name'] = args.eval_model
        elif 'scorer' in cfg:
            cfg['scorer']['model_name'] = args.eval_model
            
    if args.opt_model:
        cfg['optimizer']['model_name'] = args.opt_model
        
    if args.iterative:
        cfg['bake']['iterative'] = True
    if args.iterative_count:
        cfg['bake']['iterative_prompt_count'] = args.iterative_count

    # 3. 建立輸出目錄
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    config_snapshot_path = os.path.join(args.output_dir, "experiment_config.yaml")
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    # 4. 路徑重導
    if 'paths' not in cfg:
        cfg['paths'] = {}

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

    # 5. 載入資源
    scorer_cfg = cfg.get('evaluation', cfg.get('scorer'))
    scorer = LLMClient(scorer_cfg, role='scorer', pricing=scorer_cfg.get('pricing', {}))
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer']['pricing'])

    if hasattr(config_loader, 'load_meta_prompts'):
        meta_prompts = config_loader.load_meta_prompts(cfg['paths'].get('meta_prompt_dir', 'meta_prompt'))
    else:
        meta_prompts = {}
        mp_dir = cfg['paths'].get('meta_prompt_dir', 'meta_prompt')
        if os.path.exists(mp_dir):
            for f in os.listdir(mp_dir):
                if f.endswith('.txt'):
                    with open(os.path.join(mp_dir, f), 'r', encoding='utf-8') as file:
                        meta_prompts[f.replace('.txt', '')] = file.read()

    # 6. 載入資料
    print(f"📚 Loading Dataset [{active_task}]...")
    dataset = data_loader.load_specific_dataset(active_task, task_cfg)
    print(f"   - Loaded {len(dataset)} samples.")

    # 7. 啟動 Engine
    print("🚀 Starting BAKE Ablation Study (Mode: Success-Only Rule Extraction)")
    
    engine = SuccessOnlyBakeEngine(scorer, optimizer, cfg, meta_prompts)
    
    try:
        final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])
        
        # (A) 儲存 TXT
        with open(cfg['paths']['output_file'], "w", encoding="utf-8") as f:
            f.write("\n".join(final_prompts))
        
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

        # 儲存 Rule
        rule_path = os.path.join(args.output_dir, "final_rule.txt")
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(final_rule)
        
        scorer.save_cost_record(cfg['paths']['cost_log'])
        optimizer.save_cost_record(cfg['paths']['cost_log'])
        
        print(f"\n✅ Ablation Study Completed Successfully!")
        print(f"   Saved to: {args.output_dir}")

    except Exception as e:
        print(f"\n❌ Experiment Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()