import os
import re
import argparse
import sys
import json
import time
import requests  # 用於呼叫 API
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
#  GLOBAL CONFIGURATION (預設值)
# ==============================================================================
CONF_API_URL = "http://140.113.86.14:11434/api/chat"
CONF_MODEL_NAME = "qwen2.5:14b"
CONF_TEMPERATURE = 0.0
CONF_DEFAULT_LIMIT = 300 #[修正] 預設為 -1，代表跑全部資料
CONF_DEFAULT_SPLIT = "validation"

CONF_DEFAULT_SUBJECTS = [         
    "high_school_mathematics",
    "high_school_world_history",
    "high_school_physics",
    "professional_law",
    "business_ethics"
]

class LLMClient:
    def __init__(self, config, role, pricing):
        self.config = config
        self.api_url = config.get("base_url", CONF_API_URL)
        self.model_name = config.get("model_name", CONF_MODEL_NAME)
        self.temperature = config.get("temperature", CONF_TEMPERATURE)

    def chat(self, system_prompt, user_prompt):
        """
        使用 requests 呼叫 Ollama 原生 API (/api/chat)
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9 
            },
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            if response.status_code != 200:
                print(f"⚠️ API Error ({response.status_code}): {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            else:
                print(f"⚠️ Invalid format: {result}")
                return ""
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return ""

# ==========================================
#  Helper Functions
# ==========================================

def to_float_maybe(s: str) -> float:
    if not s: raise ValueError
    matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
    if matches: return float(matches[-1])
    raise ValueError

def extract_choice(s: str) -> str:
    if not s: raise ValueError
    pattern = r"(?:Answer|Option|Choice)?\s*[:\-\s]*\(?([A-D])\)?"
    matches = re.findall(pattern, s, re.IGNORECASE)
    if matches: return matches[-1].upper()
    clean_s = s.strip()
    if len(clean_s) < 5 and clean_s.upper() in ['A', 'B', 'C', 'D']:
        return clean_s.upper()
    raise ValueError(f"No choice found in: {s}")

def validate_answer(prediction: str, ground_truth: str, task_type: str) -> bool:
    try:
        if task_type == "math":
            return abs(to_float_maybe(prediction) - to_float_maybe(ground_truth)) < 1e-6
        elif task_type == "multiple_choice":
            return extract_choice(prediction) == ground_truth.upper().strip()
        else:
            return prediction.strip() == ground_truth.strip()
    except ValueError:
        return False

def file_has_content(filepath: str) -> bool:
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

# ==========================================
#  Data Loading (修正重點)
# ==========================================

def load_mmlu_data(subjects, limit, split):
    data = []
    # 這裡的 split (如 "validation") 只是告訴 HuggingFace 要讀哪一個檔案，
    # 不是把資料切一半的意思。我們讀進來後會全部使用。
    print(f"Loading MMLU data (File Split: {split})...")
    
    for sub in subjects:
        try:
            ds = load_dataset("cais/mmlu", sub, split=split)
            
            # [修正] 判斷是否跑全量資料
            if limit is not None and limit > 0:
                print(f"  - {sub}: Loading first {limit} samples (Limit set)")
                selected = list(ds)[:limit]
            else:
                print(f"  - {sub}: Loading ALL {len(ds)} samples (Full Evaluation)")
                selected = list(ds) # 讀取全部

            options_map = ["A", "B", "C", "D"]
            for item in selected:
                formatted_q = f"{item['question']}\n"
                for opt, content in zip(["A", "B", "C", "D"], item['choices']):
                    formatted_q += f"({opt}) {content}\n"
                formatted_q += "Answer:"
                data.append({
                    "question": formatted_q,
                    "ground_truth": options_map[item['answer']],
                    "type": "multiple_choice",
                    "subject": sub
                })
        except Exception as e:
            print(f"Failed to load subject '{sub}': {e}")
            
    print(f"Total samples loaded for evaluation: {len(data)}")
    return data

def parse_optimized_prompts(filepath):
    prompts = []
    if not file_has_content(filepath):
        return prompts
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            match = re.match(r'^\\s*(.*)', line)
            if match: prompts.append(match.group(1))
            else: prompts.append(line)
    return prompts

# ==========================================
#  Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--model", type=str, default=CONF_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=CONF_DEFAULT_LIMIT)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--subjects", nargs="+", default=CONF_DEFAULT_SUBJECTS)
    parser.add_argument("--split", type=str, default=CONF_DEFAULT_SPLIT)
    parser.add_argument("--api_url", type=str, default=CONF_API_URL)
    args = parser.parse_args()

    folder_name = os.path.basename(os.path.normpath(args.folder))
    prompts_path = os.path.join(args.folder, "optimized_prompts.txt")
    
    config = {
        "base_url": args.api_url,
        "model_name": args.model,
        "temperature": CONF_TEMPERATURE
    }
    client = LLMClient(config, role="evaluator", pricing={})
    
    prompts = parse_optimized_prompts(prompts_path)
    if not prompts:
        print("No prompts found.")
        return

    test_data = load_mmlu_data(subjects=args.subjects, limit=args.limit, split=args.split)
    if not test_data: return

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{folder_name}_{args.split}_eval.txt")

    print(f"\nStarting evaluation for: {folder_name}")
    print(f"Server: {args.api_url}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(test_data)} (Full Set: {'Yes' if args.limit < 0 else 'No'})")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"Evaluation Report\nModel: {args.model}\nData Split: {args.split} (Full)\nSubjects: {args.subjects}\n\n")
        for i, prompt_text in enumerate(prompts):
            print(f"\nTesting Prompt #{i+1}...")
            correct_count = 0
            total = len(test_data)
            for item in tqdm(test_data, desc=f"Prompt {i+1}"):
                prediction = client.chat(prompt_text, item['question'])
                if validate_answer(prediction, item['ground_truth'], item['type']):
                    correct_count += 1
            
            accuracy = (correct_count / total) * 100
            result_str = f"Prompt #{i+1} Accuracy: {accuracy:.2f}% ({correct_count}/{total})\nPrompt: {prompt_text[:100]}...\n"
            print(result_str.strip())
            f_out.write(result_str + "-"*30 + "\n")

if __name__ == "__main__":
    main()