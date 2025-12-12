from datasets import load_dataset

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
            selected = list(ds)[offset : offset + limit]
            for item in selected:
                mixed_data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": "math",
                    "source": "gsm8k"
                })

        elif name == "mmlu":
            # [關鍵修改] 讀取 subsets 列表
            target_subsets = cfg.get('subsets', [])
            
            # 如果使用者只填了單個 string (相容舊設定)，轉成 list
            if isinstance(target_subsets, str):
                target_subsets = [target_subsets]
                
            # 如果沒填，給一個預設值
            if not target_subsets:
                target_subsets = ["high_school_mathematics"]

            print(f"[DataLoader] Loading MMLU subsets: {target_subsets} (limit per subset={limit})...")

            for sub in target_subsets:
                try:
                    # 載入特定子集
                    ds = load_dataset("cais/mmlu", sub, split=split)
                    selected = list(ds)[offset : offset + limit]
                    
                    options_map = ["A", "B", "C", "D"]
                    for item in selected:
                        q_text = format_mmlu_question(item['question'], item['choices'])
                        a_text = options_map[item['answer']]
                        
                        mixed_data.append({
                            "question": q_text,
                            "answer": a_text,
                            "type": "multiple_choice",
                            "source": f"mmlu_{sub}" # 標記來源，讓 Log 更清楚
                        })
                except Exception as e:
                    print(f"  [Warn] Failed to load MMLU subset '{sub}': {e}")
    
    print(f"[DataLoader] Total samples loaded: {len(mixed_data)}")
    return mixed_data