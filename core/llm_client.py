import os
import csv
import time
from typing import Dict, Any
from openai import OpenAI

class LLMClient:
    def __init__(self, config: Dict[str, Any], role: str, pricing: Dict[str, float]):
        """
        :param role: 'scorer' or 'optimizer' (用於區分價格)
        """
        self.role = role
        self.config = config
        self.input_price = pricing.get('input', 0.0)
        self.output_price = pricing.get('output', 0.0)
        
        # Token 累計器
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # 初始化 Client
        self._init_client()

    def _init_client(self):
        provider = self.config.get("provider", "openai")
        api_key = self.config.get("api_key", os.getenv("OPENAI_API_KEY"))
        base_url = self.config.get("base_url", None)
        
        if provider == "ollama":
            if not base_url: base_url = "http://localhost:11434/v1"
            if not api_key: api_key = "ollama"
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = self.config.get("model_name", "gpt-3.5-turbo")


    def chat(self, system_prompt: str, user_prompt: str) -> str:
        # [修正] 移除 try-except，讓 OpenAI/Network 相關的異常直接拋出
        # 這樣上層 (bake_engine) 才能區分是「連線失敗」還是「模型回答為空」
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 512)
        )
        
        # 自動計費
        if response.usage:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens
            self.usage["completion_tokens"] += response.usage.completion_tokens
            self.usage["total_tokens"] += response.usage.total_tokens
        
        return response.choices[0].message.content.strip()


    def get_cost(self) -> float:
        """計算當前累積金額 (USD)"""
        in_cost = (self.usage["prompt_tokens"] / 1_000_000) * self.input_price
        out_cost = (self.usage["completion_tokens"] / 1_000_000) * self.output_price
        return in_cost + out_cost

    def save_cost_record(self, filepath: str):
        """將成本寫入 CSV 檔案"""
        file_exists = os.path.isfile(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Role", "Model", "Input Tokens", "Output Tokens", "Total Tokens", "Est. Cost ($)"])
            
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self.role,
                self.model_name,
                self.usage["prompt_tokens"],
                self.usage["completion_tokens"],
                self.usage["total_tokens"],
                f"{self.get_cost():.5f}"
            ])