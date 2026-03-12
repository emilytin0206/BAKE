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
    解析設定檔中的 subsets，支援 "all"、領域名稱 (如 "social_sciences") 
    或個別子集清單。
    """
    # 定義 MMLU 四大領域對應的子集
    MMLU_CATEGORIES = {
        "stem": [
            "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics", "computer_security",
            "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
            "high_school_physics", "machine_learning", "statistics"
        ],
        "humanities": [
            "formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history",
            "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
            "philosophy", "prehistory", "professional_law", "world_religions"
        ],
        "social_sciences": [
            "econometrics", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
            "human_sexuality", "professional_psychology", "public_relations", "sociology"
        ],
        "other": [
            "business_ethics", "clinical_knowledge", "college_medicine", "dietetics", "global_facts",
            "management", "marketing", "medical_genetics", "miscellaneous", "nutrition",
            "professional_accounting", "professional_medicine", "security_studies", "us_foreign_policy", "virology"
        ]
    }

    target_subsets = config.get('subsets', [])
    
    # 將輸入統一轉為 list 處理
    if isinstance(target_subsets, str):
        target_subsets = [target_subsets]
        
    resolved_list = []
    
    for item in target_subsets:
        item_lower = item.lower()
        # 1. 處理 "all"
        if item_lower == "all":
            try:
                print("  ↳ [Resolver] Detected 'all'. Fetching MMLU configs...")
                all_configs = get_dataset_config_names("cais/mmlu")
                return [c for c in all_configs if c not in ["all", "auxiliary_train"]]
            except Exception as e:
                print(f"  [Error] Failed to fetch MMLU configs: {e}")
                return ["high_school_mathematics"]
        
        # 2. 處理領域關鍵字 (如 social_sciences)
        elif item_lower in MMLU_CATEGORIES:
            print(f"  ↳ [Resolver] Detected category: {item_lower}")
            resolved_list.extend(MMLU_CATEGORIES[item_lower])
        
        # 3. 處理個別子集名稱
        else:
            resolved_list.append(item)

    # 預設回傳
    if not resolved_list:
        resolved_list = ["high_school_mathematics"]
        
    return resolved_list

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
        # 解析子集 (支援領域關鍵字)
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
    
    if do_shuffle:
        print("[DataLoader] 🔀 Shuffling all samples (Mixed Mode)...")
        random.seed(42) 
        random.shuffle(data_list)
    else:
        print("[DataLoader] ⬇️ Keeping original sequential order (Sequential Mode).")
        
    return data_list