import yaml
import os

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_meta_prompts(directory):
    prompts = {}
    if not os.path.exists(directory): return {}
    for f in os.listdir(directory):
        if f.endswith(".txt"):
            with open(os.path.join(directory, f), "r", encoding="utf-8") as file:
                prompts[f.replace(".txt", "")] = file.read().strip()
    return prompts