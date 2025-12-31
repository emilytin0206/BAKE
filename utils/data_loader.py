# utils/data_loader.py

from datasets import load_dataset, get_dataset_config_names
import random 

def format_mmlu_question(question, choices):
    options = ["A", "B", "C", "D"]
    formatted = f"{question}\n"
    for opt, content in zip(options, choices):
        formatted += f"({opt}) {content}\n"
    formatted += "Answer:" 
    return formatted

def resolve_mmlu_subsets(config):
    """
    解析設定檔中的 subsets，處理 "all" 的情況並回傳完整的子集列表。
    """
    target_subsets = config.get('subsets', [])
    
    if isinstance(target_subsets, str):
        target_subsets = [target_subsets]
        
    # 處理 "all"
    if "all" in target_subsets:
        try:
            print("  ↳ [Resolver] Detected 'all'. Fetching MMLU configs...")
            all_configs = get_dataset_config_names("cais/mmlu")
            # 排除非題目的 config
            target_subsets = [c for c in all_configs if c not in ["all", "auxiliary_train"]]
        except Exception as e:
            print(f"  [Error] Failed to fetch MMLU configs: {e}")
            target_subsets = ["high_school_mathematics"] # Fallback

    if not target_subsets:
        target_subsets = ["high_school_mathematics"]
        
    return target_subsets

def load_specific_dataset(task_name, config):
    """
    根據 active_task 與其 config 載入資料
    config 中可包含 'shuffle': True/False
    """
    data_list = []
    limit = config.get('limit', 10)
    offset = config.get('offset', 0)
    split = config.get('split', 'train')
    do_shuffle = config.get('shuffle', False) 

    print(f"[DataLoader] Loading Task: {task_name} (Split: {split}, Limit: {limit}, Shuffle: {do_shuffle})")

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
        # 解析子集 (處理 'all' 或多個子集)
        target_subsets = resolve_mmlu_subsets(config)
            
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
    
    # 根據設定決定是否打散
    if do_shuffle:
        print("[DataLoader] 🔀 Shuffling all samples (Mixed Mode)...")
        random.seed(42) 
        random.shuffle(data_list)
    else:
        print("[DataLoader] ⬇️ Keeping original sequential order (Sequential Mode).")
        
    return data_list