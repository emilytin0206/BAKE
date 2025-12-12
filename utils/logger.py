import json
import os
import threading

# [修正] 建立全域鎖 (Global Lock)
_lock = threading.Lock()

def init_files(filepaths):
    with _lock:  # 初始化時也加上鎖比較保險
        for fp in filepaths:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, 'w').close() # Clear file

def log_jsonl(filepath, data):
    # [修正] 使用 with _lock 確保寫入原子性
    with _lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def log_rule(filepath, title, content):
    # [修正] 使用 with _lock
    with _lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*10} {title} {'='*10}\n{content}\n")