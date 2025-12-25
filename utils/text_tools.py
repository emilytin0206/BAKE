import re
import os

def to_float_maybe(s: str) -> float:
    if not s: raise ValueError
    matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
    if matches: return float(matches[-1])
    raise ValueError

def extract_choice(s: str) -> str:
    """
    從模型輸出中提取選擇題答案 (A/B/C/D/E)。
    
    策略流程：
    1. [最高優先] LaTeX Boxed: 尋找 \boxed{A}，這是最精確的格式。
    2. [結論定位] 關鍵字切割: 尋找 "Answer is" 等詞，只保留其後的內容作為「結論區」。
    3. [智慧提取] 
       - 若有找到關鍵字 (鎖定結論區): 取區域內的「第一個」選項 (避免抓到後面補充說明的錯誤選項)。
       - 若無找到關鍵字 (全文搜尋): 取全文的「最後一個」選項 (假設結論在最後)。
    4. [保底] 極簡短字串處理。
    """
    if not s: raise ValueError("Empty input string")
    
    # 1. 預處理
    text = s.strip()

    # 2. [最強優先級] LaTeX Boxed 格式: \boxed{A}
    match_boxed = re.search(r'\\boxed\{\s*([A-E])\s*\}', text, re.IGNORECASE)
    if match_boxed:
        return match_boxed.group(1).upper()

    # 3. [關鍵邏輯] 標準化與切割 (定位結論區)
    text_lower = text.lower()
    keywords = ['answer is', 'answer:', 'the answer is', 'correct answer is', 'option:', 'choice:']
    
    found_keyword = False
    for pat in keywords:
        if pat in text_lower:
            # 使用 rsplit 確保我們抓的是最後一次出現的關鍵字 (例如文中有多次 "Answer:")
            # 取 [-1] 代表取關鍵字「後面」的內容
            text_lower = text_lower.rsplit(pat, 1)[-1].strip()
            found_keyword = True 
            break
            
    # 4. [提取選項] 根據是否鎖定結論區，決定抓頭還是抓尾
    
    # 4.1 尋找括號格式: (A), (B)
    matches_paren = re.findall(r'\(([A-E])\)', text_lower, re.IGNORECASE)
    if matches_paren:
        # 如果有鎖定結論區 -> 答案通常在開頭 -> 取第一個
        # 如果沒鎖定 (全文) -> 答案通常在結尾 -> 取最後一個
        return matches_paren[0].upper() if found_keyword else matches_paren[-1].upper()
        
    # 4.2 尋找單獨字母: A, B (需有邊界 \b，避免抓到單字裡的字母)
    matches_word = re.findall(r'\b([A-E])\b', text_lower, re.IGNORECASE)
    if matches_word:
        return matches_word[0].upper() if found_keyword else matches_word[-1].upper()

    # 5. [保底策略] 極簡字串處理
    # 如果上面都沒抓到，但字串原本就很短 (例如直接輸出 "A." 或 "B")
    # 注意：這裡回頭看原始 s 的長度，避免被切割後誤判
    if len(s.strip()) < 10:
        match_simple = re.search(r'([A-E])', s, re.IGNORECASE)
        if match_simple:
            return match_simple.group(1).upper()

    # 若真的什麼都沒抓到
    raise ValueError(f"No choice found in: {s}")

def validate_answer(prediction: str, ground_truth: str, task_type: str) -> bool:
    try:
        # 統一轉成字串並清理，避免型別錯誤
        ground_truth = str(ground_truth).strip()
        
        if task_type == "math":
            pred_val = to_float_maybe(prediction)
            gt_val = to_float_maybe(ground_truth)
            return abs(pred_val - gt_val) < 1e-6
        elif task_type == "multiple_choice":
            # 使用更新後的提取邏輯
            pred_choice = extract_choice(prediction)
            return pred_choice == ground_truth.upper()
        else:
            return prediction.strip() == ground_truth
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

def file_has_content(filepath: str) -> bool:
    if not os.path.exists(filepath):
        return False
    return os.path.getsize(filepath) > 0