from datasets import load_dataset, get_dataset_config_names

def format_mmlu_question(question, choices):
    options = ["A", "B", "C", "D"]
    formatted = f"{question}\n"
    for opt, content in zip(options, choices):
        formatted += f"({opt}) {content}\n"
    formatted += "Answer:" 
    return formatted

def load_mixed_datasets(datasets_config):
    mixed_data = []
    
    for cfg in datasets_config:
        name = cfg['name']
        limit = cfg.get('limit', 10)
        offset = cfg.get('offset', 0)
        split = cfg.get('split', 'train')
        
        if name == "gsm8k":
            print(f"[DataLoader] Loading GSM8K (limit={limit})...")
            ds = load_dataset("gsm8k", "main", split=split)
            if limit > 0:
                selected = list(ds)[offset : offset + limit]
            else:
                selected = list(ds)[offset:]
                
            for item in selected:
                mixed_data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": "math",
                    "source": "gsm8k"
                })

        elif name == "mmlu":
            target_subsets = cfg.get('subsets', [])
            
            if isinstance(target_subsets, str):
                target_subsets = [target_subsets]
                
            # [新增功能] 如果設定為 "all"，自動抓取所有 MMLU 子集名稱
            if "all" in target_subsets:
                print("[DataLoader] Detected 'all' in subsets. Fetching all MMLU configs...")
                try:
                    all_configs = get_dataset_config_names("cais/mmlu")
                    # 過濾掉 'all' (總集) 和 'auxiliary_train' (輔助訓練集)，只保留各個學科
                    target_subsets = [c for c in all_configs if c not in ["all", "auxiliary_train"]]
                except Exception as e:
                    print(f"  [Error] Failed to fetch MMLU configs: {e}")
                    target_subsets = ["high_school_mathematics"] # Fallback

            if not target_subsets:
                target_subsets = ["high_school_mathematics"]

            print(f"[DataLoader] Loading MMLU subsets (Total: {len(target_subsets)})...")
            if limit <= 0:
                print(f"  ↳ Limit set to {limit}, loading ALL data per subset.")
            else:
                print(f"  ↳ Limit set to {limit} per subset.")

            for sub in target_subsets:
                try:
                    ds = load_dataset("cais/mmlu", sub, split=split)
                    
                    # [關鍵修改] 判斷是否讀取全部資料
                    if limit > 0:
                        selected = list(ds)[offset : offset + limit]
                    else:
                        selected = list(ds)[offset:] # 讀取全部
                    
                    options_map = ["A", "B", "C", "D"]
                    for item in selected:
                        q_text = format_mmlu_question(item['question'], item['choices'])
                        a_text = options_map[item['answer']]
                        
                        mixed_data.append({
                            "question": q_text,
                            "answer": a_text,
                            "type": "multiple_choice",
                            "source": f"mmlu_{sub}"
                        })
                except Exception as e:
                    print(f"  [Warn] Failed to load MMLU subset '{sub}': {e}")
    
    print(f"[DataLoader] Total samples loaded: {len(mixed_data)}")
    return mixed_data