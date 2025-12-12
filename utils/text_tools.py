import re

def to_float_maybe(s: str) -> float:
    if not s: raise ValueError
    matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
    if matches: return float(matches[-1])
    raise ValueError

def extract_choice(s: str) -> str:
    """
    從模型輸出中提取 A, B, C, D。
    支援格式: "The answer is (A)", "Option C", "Answer: B", 或直接 "D"
    """
    if not s: raise ValueError
    
    # 1. 嘗試找最後出現的明確答案格式 (如 "Answer: (A)")
    # regex 解釋: 找 Answer/Option 關鍵字，後面接 A-D，忽略大小寫
    pattern = r"(?:Answer|Option|Choice)?\s*[:\-\s]*\(?([A-D])\)?"
    matches = re.findall(pattern, s, re.IGNORECASE)
    
    if matches:
        return matches[-1].upper() # 回傳最後一個匹配到的
    
    # 2. 保底策略：如果內容極短 (例如只回 "A")，直接判定
    clean_s = s.strip()
    if len(clean_s) < 5 and clean_s.upper() in ['A', 'B', 'C', 'D']:
        return clean_s.upper()

    raise ValueError(f"No choice found in: {s}")

# === [NEW] 統一驗證入口 ===
def validate_answer(prediction: str, ground_truth: str, task_type: str) -> bool:
    """
    根據 task_type 自動切換驗證邏輯
    """
    try:
        if task_type == "math":
            # 數學題：比對數值 (容許浮點誤差)
            pred_val = to_float_maybe(prediction)
            gt_val = to_float_maybe(ground_truth)
            return abs(pred_val - gt_val) < 1e-6
            
        elif task_type == "multiple_choice":
            # 選擇題：比對字母 (A, B, C, D)
            pred_choice = extract_choice(prediction)
            # Ground Truth 通常是 "A", "B"... 或是 index 0, 1... (Loader 需處理)
            return pred_choice == ground_truth.upper().strip()
            
        else:
            # 預設：字串完全相等
            return prediction.strip() == ground_truth.strip()
            
    except ValueError:
        return False

def extract_tags(text: str, tag_name: str) -> list:
    pattern = f"<{tag_name}_BEGIN>(.*?)</{tag_name}_END>"
    return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]

def insert_prompts_template(correct, wrong):
    c = "\n".join(correct) if correct else "None"
    w = "\n".join(wrong) if wrong else "None"
    return f"Correct:\n{c}\n---\nWrong:\n{w}"