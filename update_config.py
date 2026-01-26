# update_config.py
import yaml
import sys

# 定義 GPT-4o 和 Gemini 的 Prompts 資料
PROMPT_SETS = {
    "gpt4o": [
        "Restate the task briefly, decompose it into minimal logical steps, solve each step, and produce a result that directly satisfies the goal.",
        "Solve the task with explicit reasoning, verifying correctness after each major step and revising any inconsistent logic before proceeding.",
        "Identify necessary assumptions, avoid unstated ones, reason only from given information, and choose the most consistent interpretation.",
        "Generate multiple solution strategies, compare them for correctness and efficiency, select the strongest, and output only the final result.",
        "Prioritize clarity and precision, keep reasoning concise, remove unnecessary complexity, and ensure the output fully meets the task."
    ],
    "gemini": [
        # 1. The Intent Decoder
        "Identify the core intent and underlying goal of this request, then provide a direct, high-quality response that best satisfies that goal.",
        # 2. The Internal Critic
        "Generate a preliminary response internally, critique it for clarity, accuracy, and completeness, and then output only the improved final version.",
        # 3. The Logical Deconstructor
        "Break this task down into its fundamental components, address each component logically, and synthesize them into a coherent solution.",
        # 4. The Dynamic Adapter
        "Adopt the most effective perspective and tone for this specific task to maximize the helpfulness and relevance of the response.",
        # 5. The Structured Synthesizer
        "Structure the response clearly with a logical flow, prioritizing key information first, followed by necessary details or context."
    ]
}

def update_config(source_key, config_path="config.yaml"):
    if source_key not in PROMPT_SETS:
        print(f"Error: Unknown prompt source '{source_key}'")
        sys.exit(1)

    prompts = PROMPT_SETS[source_key]

    try:
        # 讀取現有 Config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        # 更新 initial_prompts
        config['initial_prompts'] = prompts
        
        # 寫回 Config (保留 YAML 格式)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
        print(f"✅ Successfully updated config.yaml with {len(prompts)} prompts from [{source_key}]")
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_config.py <gpt4o|gemini>")
        sys.exit(1)
    
    update_config(sys.argv[1])