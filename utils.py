import os
import re
import yaml
import time
import json
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset

# 設定 Log
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===========================
# LLM Client Wrapper
# ===========================
class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get("provider", "openai").lower()
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        
        api_key = config.get("api_key", os.getenv("OPENAI_API_KEY"))
        base_url = config.get("base_url", None)

        if self.provider == "ollama":
            if not base_url: base_url = "http://localhost:11434/v1"
            if not api_key: api_key = "ollama"
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """同步呼叫，主要給 Optimizer 用"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error calling {self.model_name}: {e}")
            return ""

# ===========================
# Helper Functions (Logic Ported)
# ===========================

def to_float_maybe(s: str) -> float:
    """嘗試從字串提取浮點數 (針對 GSM8K)"""
    if s is None: raise ValueError("None result")
    # 移除逗號 (1,000 -> 1000)
    s_clean = s.replace(',', '').strip()
    # 嘗試提取最後一個數字
    matches = re.findall(r'-?\d+\.?\d*', s_clean)
    if matches:
        return float(matches[-1])
    raise ValueError(f"No float found in: {s}")

def extract_content(text: str, tag_start: str, tag_end: str) -> List[str]:
    """從 Tag 中提取內容 (支援多行 DOTALL)"""
    pattern = re.escape(tag_start) + r"(.*?)" + re.escape(tag_end)
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]

def insert_prompts_to_content(template: str, correct: List[str], wrong: List[str]) -> str:
    """將 Prompt 插入到模板中 (模擬你的 insert_prompts_to_content)"""
    c_block = "\n".join(correct) if correct else "None"
    w_block = "\n".join(wrong) if wrong else "None"
    
    # 這裡假設 MetaPrompt 使用了 {correct_prompts} 和 {wrong_prompts} 作為 placeholder
    # 或是依據你的舊代碼 replace 特定 tag
    text = template.replace("###CORRECT_PROMPT_BLOCK###", c_block)
    text = text.replace("###INCORRECT_PROMPT_BLOCK###", w_block)
    
    # 相容 format 語法
    try:
        text = text.format(correct_prompts=c_block, wrong_prompts=w_block)
    except:
        pass
    return text

# ===========================
# Config & Data
# ===========================
def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_meta_prompts(directory: str) -> Dict[str, str]:
    prompts = {}
    files = ["analyze_only.txt", "rewrite_from_analysis.txt", 
             "rule_summarization.txt", "combine_rules.txt", "prompt_generation.txt"]
    
    if not os.path.exists(directory):
        return {}
        
    for filename in files:
        key = filename.replace(".txt", "")
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                prompts[key] = f.read().strip()
    return prompts

def load_data(dataset_name: str, split: str = "train", offset: int = 0, limit: int = 20):
    data = []
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        selected = list(ds)[offset : offset + limit]
        for item in selected:
            data.append({"question": item["question"], "answer": item["answer"]})
    return data