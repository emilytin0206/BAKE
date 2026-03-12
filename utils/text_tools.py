# import re
# import os

# def to_float_maybe(s: str) -> float:
#     if not s: raise ValueError
#     matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
#     if matches: return float(matches[-1])
#     raise ValueError

# def extract_choice(s: str) -> str:
#     """
#     從模型輸出中提取選擇題答案 (A/B/C/D/E)。
    
#     策略流程：
#     1. [最高優先] LaTeX Boxed: 尋找 \boxed{A}，這是最精確的格式。
#     2. [結論定位] 關鍵字切割: 尋找 "Answer is" 等詞，只保留其後的內容作為「結論區」。
#     3. [智慧提取] 
#        - 若有找到關鍵字 (鎖定結論區): 取區域內的「第一個」選項 (避免抓到後面補充說明的錯誤選項)。
#        - 若無找到關鍵字 (全文搜尋): 取全文的「最後一個」選項 (假設結論在最後)。
#     4. [保底] 極簡短字串處理。
#     """
#     if not s: raise ValueError("Empty input string")
    
#     # 1. 預處理
#     text = s.strip()

#     # 2. [最強優先級] LaTeX Boxed 格式: \boxed{A}
#     match_boxed = re.search(r'\\boxed\{\s*([A-E])\s*\}', text, re.IGNORECASE)
#     if match_boxed:
#         return match_boxed.group(1).upper()

#     # 3. [關鍵邏輯] 標準化與切割 (定位結論區)
#     text_lower = text.lower()
#     keywords = ['answer is', 'answer:', 'the answer is', 'correct answer is', 'option:', 'choice:']
    
#     found_keyword = False
#     for pat in keywords:
#         if pat in text_lower:
#             # 使用 rsplit 確保我們抓的是最後一次出現的關鍵字 (例如文中有多次 "Answer:")
#             # 取 [-1] 代表取關鍵字「後面」的內容
#             text_lower = text_lower.rsplit(pat, 1)[-1].strip()
#             found_keyword = True 
#             break
            
#     # 4. [提取選項] 根據是否鎖定結論區，決定抓頭還是抓尾
    
#     # 4.1 尋找括號格式: (A), (B)
#     matches_paren = re.findall(r'\(([A-E])\)', text_lower, re.IGNORECASE)
#     if matches_paren:
#         # 如果有鎖定結論區 -> 答案通常在開頭 -> 取第一個
#         # 如果沒鎖定 (全文) -> 答案通常在結尾 -> 取最後一個
#         return matches_paren[0].upper() if found_keyword else matches_paren[-1].upper()
        
#     # 4.2 尋找單獨字母: A, B (需有邊界 \b，避免抓到單字裡的字母)
#     matches_word = re.findall(r'\b([A-E])\b', text_lower, re.IGNORECASE)
#     if matches_word:
#         return matches_word[0].upper() if found_keyword else matches_word[-1].upper()

#     # 5. [保底策略] 極簡字串處理
#     # 如果上面都沒抓到，但字串原本就很短 (例如直接輸出 "A." 或 "B")
#     # 注意：這裡回頭看原始 s 的長度，避免被切割後誤判
#     if len(s.strip()) < 10:
#         match_simple = re.search(r'([A-E])', s, re.IGNORECASE)
#         if match_simple:
#             return match_simple.group(1).upper()

#     # 若真的什麼都沒抓到
#     raise ValueError(f"No choice found in: {s}")

# def validate_answer(prediction: str, ground_truth: str, task_type: str) -> bool:
#     try:
#         # 統一轉成字串並清理，避免型別錯誤
#         ground_truth = str(ground_truth).strip()
        
#         if task_type == "math":
#             pred_val = to_float_maybe(prediction)
#             gt_val = to_float_maybe(ground_truth)
#             return abs(pred_val - gt_val) < 1e-6
#         elif task_type == "multiple_choice":
#             # 使用更新後的提取邏輯
#             pred_choice = extract_choice(prediction)
#             return pred_choice == ground_truth.upper()
#         else:
#             return prediction.strip() == ground_truth
#     except ValueError:
#         return False

# def extract_tags(text: str, tag_name: str) -> list:
#     if not text: return []
    
#     # 1. 標準格式 <TAG_BEGIN>...</TAG_END> (忽略大小寫)
#     pattern = f"<{tag_name}_BEGIN>(.*?)</{tag_name}_END>"
#     matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
#     # 2. 如果沒抓到，嘗試容錯格式 (例如中間是空白 <TAG BEGIN>)
#     if not matches:
#         pattern_loose = f"<{tag_name}[ _]BEGIN>(.*?)</{tag_name}[ _]END>"
#         matches = re.findall(pattern_loose, text, re.DOTALL | re.IGNORECASE)
        
#     return [m.strip() for m in matches]

# def insert_prompts_template(correct, wrong):
#     c = "\n".join(correct) if correct else "None"
#     w = "\n".join(wrong) if wrong else "None"
#     return f"Correct:\n{c}\n---\nWrong:\n{w}"

# def file_has_content(filepath: str) -> bool:
#     if not os.path.exists(filepath):
#         return False
#     return os.path.getsize(filepath) > 0


import re
import os
import string

def _get_answer_text(input_text: str, answer_symbol: str) -> str:
    """
    從題目中提取選項文字 (例如提取 '(A) Paris' 中的 'Paris')
    對應官方 OPRO eval_utils.py 中的 _get_answer_text
    """
    try:
        # 尋找特定選項，並擷取文字直到遇到下一個選項或結尾
        pattern = rf"\({answer_symbol.upper()}\)\s*(.*?)(?=\([A-Za-z]\)|What's the answer|$)"
        match = re.search(pattern, input_text, re.DOTALL)
        if match:
            return match.group(1).strip().lower()
    except Exception:
        pass
    return ""

def validate_answer(prediction: str, ground_truth: str, task_type: str, input_text: str = "") -> bool:
    """
    完全還原官方 OPRO (eval_utils.py & metrics.py) 的評分邏輯
    """
    pred_clean = str(prediction).strip().lower()
    targ_clean = str(ground_truth).strip().lower()
    
    if task_type == "math":
        try:
            # OPRO 官方數值比較邏輯 (帶有 1e-5 容錯率)
            target_num_str = targ_clean.replace(',', '')
            target_num = float(target_num_str) if '.' in target_num_str else int(target_num_str)
            
            pred_clean_no_comma = pred_clean.replace(',', '')
            pred_nums_str = re.findall(r'-?\d*\.?\d+', pred_clean_no_comma)
            
            if pred_nums_str:
                pred_last_num = float(pred_nums_str[-1])
                if abs(pred_last_num - target_num) <= 1e-5:
                    return True
        except Exception:
            pass
        return False

    elif task_type == "multiple_choice":
        if len(targ_clean) == 1 and targ_clean in ['a', 'b', 'c', 'd', 'e']:
            
            # OPRO 官方邏輯 1: 檢查括號選項
            bracketed_letters = [f"({l})" for l in string.ascii_lowercase]
            choice_in_pred_all = [item in pred_clean for item in bracketed_letters]
            
            extracted_ans = pred_clean
            # 若且唯若剛好只有一個括號選項
            if sum(choice_in_pred_all) == 1:
                matches = re.findall(r'\([a-z]\)', pred_clean)
                if matches:
                    extracted_ans = matches[0]
            
            targ_letter = targ_clean.replace("(", "").replace(")", "")
            pred_letter = extracted_ans.replace("(", "").replace(")", "")
            pred_no_punc = pred_letter.translate(str.maketrans("", "", string.punctuation)).strip()
            
            if pred_no_punc == targ_letter:
                return True

            # OPRO 官方邏輯 2: 選項純文本排他比對 (需依賴 input_text)
            if input_text:
                true_ans_text = _get_answer_text(input_text, targ_letter)
                
                if true_ans_text and true_ans_text in pred_clean:
                    # 找出題目中所有可用的選項 (A, B, C, D)
                    available_options = re.findall(r'\(([A-Ea-e])\)', input_text)
                    other_texts = []
                    for opt in available_options:
                        if opt.lower() != targ_letter:
                            other_text = _get_answer_text(input_text, opt)
                            if other_text:
                                other_texts.append(other_text)
                    
                    # 檢查是否排他 (確信模型沒有提到其他錯誤選項的文字)
                    is_excluded = True
                    for ot in other_texts:
                        if ot in pred_clean:
                            is_excluded = False
                            break
                            
                    if is_excluded:
                        return True
        else:
            # 防呆：如果 target 不是 A~E，回歸純字串比對
            return targ_clean == pred_clean.strip()
            
        return False
        
    else:
        # 其他任務類型
        return pred_clean == targ_clean

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