import re
import os  # [新增] 用於檢查檔案

def to_float_maybe(s: str) -> float:
    if not s: raise ValueError
    matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
    if matches: return float(matches[-1])
    raise ValueError

def extract_choice(s: str) -> str:
    if not s: raise ValueError
    pattern = r"(?:Answer|Option|Choice)?\s*[:\-\s]*\(?([A-D])\)?"
    matches = re.findall(pattern, s, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    clean_s = s.strip()
    if len(clean_s) < 5 and clean_s.upper() in ['A', 'B', 'C', 'D']:
        return clean_s.upper()
    raise ValueError(f"No choice found in: {s}")

def validate_answer(prediction: str, ground_truth: str, task_type: str) -> bool:
    try:
        if task_type == "math":
            pred_val = to_float_maybe(prediction)
            gt_val = to_float_maybe(ground_truth)
            return abs(pred_val - gt_val) < 1e-6
        elif task_type == "multiple_choice":
            pred_choice = extract_choice(prediction)
            return pred_choice == ground_truth.upper().strip()
        else:
            return prediction.strip() == ground_truth.strip()
    except ValueError:
        return False

def extract_tags(text: str, tag_name: str) -> list:
    if not text: return []
    
    # 1. 標準格式 <TAG_BEGIN>...</TAG_END> (忽略大小寫)
    pattern = f"<{tag_name}_BEGIN>(.*?)</{tag_name}_END>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    # 2. 如果沒抓到，嘗試容錯格式 (例如中間是空白 <TAG BEGIN>)
    if not matches:
        pattern_loose = f"<{tag_name}[ _]BEGIN>(.*?)</{tag_name}[ _]END>"
        matches = re.findall(pattern_loose, text, re.DOTALL | re.IGNORECASE)
        
    return [m.strip() for m in matches]

def insert_prompts_template(correct, wrong):
    c = "\n".join(correct) if correct else "None"
    w = "\n".join(wrong) if wrong else "None"
    return f"Correct:\n{c}\n---\nWrong:\n{w}"

# [新增] 檢查檔案是否存在且有內容
def file_has_content(filepath: str) -> bool:
    if not os.path.exists(filepath):
        return False
    return os.path.getsize(filepath) > 0