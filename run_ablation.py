import argparse
import yaml
import os
import sys

# 引用現有的模組
from utils import config_loader, data_loader, logger
from core.llm_client import LLMClient
# 引用我們新建的 Engine
from core.bake_engine_ablation import SuccessOnlyBakeEngine 

def main():
    parser = argparse.ArgumentParser(description="BAKE Ablation Study Runner (Success-Only)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    
    # 支援與 main.py 相同的參數覆蓋
    parser.add_argument("--task", type=str, help="Override task in config")
    parser.add_argument("--subsets", type=str, help="Override subsets (comma separated)")
    parser.add_argument("--limit", type=int, help="Override dataset limit")
    parser.add_argument("--split", type=str, help="Override dataset split")
    parser.add_argument("--eval_model", type=str, help="Override evaluation model")
    parser.add_argument("--opt_model", type=str, help="Override optimizer model")
    parser.add_argument("--iterative", action='store_true', help="Enable iterative mode")
    parser.add_argument("--iterative_count", type=int, help="Number of prompts to generate in iterative mode")

    args = parser.parse_args()

    # 1. 載入設定
    print(f"🔧 Loading Config: {args.config}")
    cfg = config_loader.load_config(args.config)
    
    # 2. 處理參數覆蓋 (CLI Override)
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
        
    cfg['dataset'][active_task] = task_cfg # 寫回 Config

    if args.eval_model:
        # 相容檢查：config 可能是 'evaluation' 或 'scorer'
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

    # 3. 建立輸出目錄並儲存 Config 快照
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    config_snapshot_path = os.path.join(args.output_dir, "experiment_config.yaml")
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    # 4. 路徑重導 (重要：確保 Log 寫入正確資料夾)
    # [修正] 這裡要確保 cfg['paths'] 存在
    if 'paths' not in cfg:
        cfg['paths'] = {}

    # 定義預設檔名，避免 config 沒寫到會報錯
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

    # 5. 載入資源 (Meta Prompts & Clients)
    print("📂 Loading Meta Prompts...")
    # 檢查 config_loader 是否有 load_meta_prompts，若無則手動讀取
    if hasattr(config_loader, 'load_meta_prompts'):
        meta_prompts = config_loader.load_meta_prompts(cfg['paths'].get('meta_prompt_dir', 'meta_prompt'))
    else:
        # Fallback
        meta_prompts = {}
        mp_dir = cfg['paths'].get('meta_prompt_dir', 'meta_prompt')
        if os.path.exists(mp_dir):
            for f in os.listdir(mp_dir):
                if f.endswith('.txt'):
                    with open(os.path.join(mp_dir, f), 'r', encoding='utf-8') as file:
                        meta_prompts[f.replace('.txt', '')] = file.read()

    # 初始化 LLM Clients
    scorer_cfg = cfg.get('evaluation', cfg.get('scorer'))
    scorer = LLMClient(scorer_cfg, role='scorer', pricing=scorer_cfg.get('pricing', {}))
    optimizer = LLMClient(cfg['optimizer'], role='optimizer', pricing=cfg['optimizer']['pricing'])

    # 6. 載入資料
    print(f"📚 Loading Dataset [{active_task}]...")
    dataset = data_loader.load_specific_dataset(active_task, task_cfg)
    print(f"   - Loaded {len(dataset)} samples.")

    # 7. 啟動 Ablation Engine
    print("🚀 Starting BAKE Ablation Study (Mode: Success-Only Rule Extraction)")
    
    # 傳入正確參數
    engine = SuccessOnlyBakeEngine(scorer, optimizer, cfg, meta_prompts)
    
    try:
        final_prompts, final_rule = engine.run(dataset, cfg['initial_prompts'])
        
        # 儲存結果
        with open(cfg['paths']['output_file'], "w", encoding="utf-8") as f:
            f.write("\n".join(final_prompts))
            
        rule_path = os.path.join(args.output_dir, "final_rule.txt")
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(final_rule)
        
        # 儲存成本紀錄
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