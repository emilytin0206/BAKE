# utils/data_loader.py

from datasets import load_dataset, get_dataset_config_names

def format_mmlu_question(question, choices):
    options = ["A", "B", "C", "D"]
    formatted = f"{question}\n"
    for opt, content in zip(options, choices):
        formatted += f"({opt}) {content}\n"
    formatted += "Answer:" 
    return formatted

def load_specific_dataset(task_name, config):
    """
    根據 active_task 與其 config 載入資料
    """
    data_list = []
    limit = config.get('limit', 10)
    offset = config.get('offset', 0)
    split = config.get('split', 'train')

    print(f"[DataLoader] Loading Task: {task_name} (Split: {split}, Limit: {limit})")

    if task_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        
        if limit > 0:
            selected = list(ds)[offset : offset + limit]
        else:
            selected = list(ds)[offset:]
            
        for item in selected:
            data_list.append({
                "question": item["question"],
                "answer": item["answer"],
                "type": "math",
                "source": "gsm8k"
            })

    elif task_name == "mmlu":
        target_subsets = config.get('subsets', [])
        
        if isinstance(target_subsets, str):
            target_subsets = [target_subsets]
            
        # 處理 "all"
        if "all" in target_subsets:
            print("  ↳ Detected 'all' subsets. Fetching MMLU configs...")
            try:
                all_configs = get_dataset_config_names("cais/mmlu")
                target_subsets = [c for c in all_configs if c not in ["all", "auxiliary_train"]]
            except Exception as e:
                print(f"  [Error] Failed to fetch MMLU configs: {e}")
                target_subsets = ["high_school_mathematics"]

        if not target_subsets:
            target_subsets = ["high_school_mathematics"]

        print(f"  ↳ Loading {len(target_subsets)} subsets...")

        for sub in target_subsets:
            try:
                ds = load_dataset("cais/mmlu", sub, split=split)
                
                if limit > 0:
                    selected = list(ds)[offset : offset + limit]
                else:
                    selected = list(ds)[offset:]
                
                options_map = ["A", "B", "C", "D"]
                for item in selected:
                    q_text = format_mmlu_question(item['question'], item['choices'])
                    a_text = options_map[item['answer']]
                    
                    data_list.append({
                        "question": q_text,
                        "answer": a_text,
                        "type": "multiple_choice",
                        "source": f"mmlu_{sub}"
                    })
            except Exception as e:
                print(f"  [Warn] Failed to load subset '{sub}': {e}")

    print(f"[DataLoader] Total samples loaded: {len(data_list)}")
    return data_list