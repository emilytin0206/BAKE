import os
import re
import argparse
import sys
import json
import time
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
#  GLOBAL CONFIGURATION (Python 內部的備用預設值)
# ==============================================================================
CONF_MODEL_NAME = "qwen2.5:7b"
CONF_API_URL = "http://localhost:11434/v1"
CONF_API_KEY = "ollama"
CONF_TEMPERATURE = 0.0
CONF_MAX_TOKENS = 1024
CONF_DEFAULT_LIMIT = 10

# 預設科目 (如果在 command line 沒指定，會用這個)
CONF_DEFAULT_SUBJECTS = ["high_school_mathematics"]
# ==============================================================================

try:
    from core.llm_client import LLMClient
except ImportError:
    from openai import OpenAI
    class LLMClient:
        def __init__(self, config, role, pricing):
            self.config = config
            self.model_name = config.get("model_name", CONF_MODEL_NAME)
            base_url = config.get("base_url", CONF_API_URL)
            api_key = config.get("api_key", CONF_API_KEY)
            self.client = OpenAI(base_url=base_url, api_key=api_key)

        def chat(self, system_prompt, user_prompt):
            try:
                temp = self.config.get("temperature", CONF_TEMPERATURE)
                max_tok = self.config.get("max_tokens", CONF_MAX_TOKENS)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error in chat: {e}")
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
#  Data Loading
# ==========================================

def load_mmlu_test_data(subjects, limit):
    data = []
    print(f"Loading MMLU test data (limit={limit} per subject)...")
    print(f"Target Subjects: {subjects}")
    
    for sub in subjects:
        try:
            ds = load_dataset("cais/mmlu", sub, split="test")
            selected = list(ds)[:limit]
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
    print(f"Total test samples loaded: {len(data)}")
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
#  Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate optimized prompts on MMLU test set")
    parser.add_argument("--folder", type=str, required=True, help="Path to experiment folder")
    parser.add_argument("--model", type=str, default=CONF_MODEL_NAME, help="Ollama model name")
    parser.add_argument("--limit", type=int, default=CONF_DEFAULT_LIMIT, help="Questions per subject")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory")
    
    # [新增] 接收 subjects 列表參數 (nargs='+' 表示可接收多個字串)
    parser.add_argument("--subjects", nargs="+", default=CONF_DEFAULT_SUBJECTS, help="List of MMLU subjects to evaluate")

    args = parser.parse_args()

    folder_name = os.path.basename(os.path.normpath(args.folder))
    prompts_path = os.path.join(args.folder, "optimized_prompts.txt")
    
    # Init Client
    config = {
        "provider": "ollama", "base_url": CONF_API_URL, "api_key": CONF_API_KEY,
        "model_name": args.model, "temperature": CONF_TEMPERATURE, "max_tokens": CONF_MAX_TOKENS
    }
    client = LLMClient(config, role="evaluator", pricing={})
    
    # Load Prompts
    prompts = parse_optimized_prompts(prompts_path)
    if not prompts:
        print("No prompts found.")
        return

    # Load Data (使用 args.subjects)
    test_data = load_mmlu_test_data(subjects=args.subjects, limit=args.limit)
    if not test_data:
        print("No data loaded.")
        return

    # Eval Loop
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{folder_name}_eval.txt")

    print(f"\nStarting evaluation for: {folder_name}")
    print(f"Model: {args.model} | Temp: {CONF_TEMPERATURE}")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"Evaluation Report for {folder_name}\n")
        f_out.write(f"Model: {args.model}\n")
        f_out.write(f"Subjects: {', '.join(args.subjects)}\n")
        f_out.write("="*50 + "\n\n")

        for i, prompt_text in enumerate(prompts):
            print(f"\nTesting Prompt #{i+1}...")
            correct_count = 0
            total = len(test_data)
            
            for item in tqdm(test_data, desc=f"Prompt {i+1}"):
                prediction = client.chat(prompt_text, item['question'])
                if validate_answer(prediction, item['ground_truth'], item['type']):
                    correct_count += 1
            
            accuracy = (correct_count / total) * 100
            result_str = f"Prompt #{i+1} Accuracy: {accuracy:.2f}% ({correct_count}/{total})\nPrompt Content: {prompt_text}\n"
            print(result_str.strip())
            f_out.write(result_str + "-"*30 + "\n")

    print(f"\nEvaluation complete. Results: {output_file}")

if __name__ == "__main__":
    main()