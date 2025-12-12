import json
import os

def init_files(filepaths):
    for fp in filepaths:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, 'w').close() # Clear file

def log_jsonl(filepath, data):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def log_rule(filepath, title, content):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*10} {title} {'='*10}\n{content}\n")