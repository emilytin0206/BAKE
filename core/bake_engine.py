import time
import csv
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import text_tools, logger

class BakeEngine:
    # 移除原本寫死的 DEFAULT_SYS_MSG，改從檔案讀取或設為 fallback
    
    def __init__(self, scorer, optimizer, config, meta_prompts):
        self.scorer = scorer
        self.optimizer = optimizer
        self.config = config
        self.meta_prompts = meta_prompts
        
        self.concurrency = config['execution']['concurrency']
        self.max_retries = config['execution']['max_retries']
        self.group_size = config['bake']['group_size']
        self.enable_iterative = config['bake'].get('iterative', False)
        self.paths = config['paths']

    def evaluate_parallel(self, query: str, answer_gt: str, prompts: List[str], task_type: str):
        """Step 1: Evaluation"""
        correct, wrong = [], []
        detailed_res = {}
        failed_outputs = {}

        # 讀取 Prompt 模板
        sys_tpl = self.meta_prompts.get("evaluate_system", "You are a helpful assistant.")
        user_tpl = self.meta_prompts.get("evaluate_user", "{prompt}\n\n{query}")

        def _worker(p):
            # 填入 User Message
            full_input = user_tpl.format(prompt=p, query=query)
            for _ in range(self.max_retries):
                try:
                    raw = self.scorer.chat(sys_tpl, full_input)
                    is_correct = text_tools.validate_answer(raw, answer_gt, task_type)
                    return (p, is_correct, raw)
                except Exception:
                    time.sleep(self.config['execution'].get('retry_delay', 1.0))
            return (p, None, None)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_p = {executor.submit(_worker, p): p for p in prompts}
            for future in as_completed(future_to_p):
                p, is_correct, raw_output = future.result()
                if is_correct is None: continue 
                
                detailed_res[p] = is_correct
                if is_correct:
                    correct.append(p)
                else:
                    wrong.append(p)
                    failed_outputs[p] = raw_output

        return correct, wrong, detailed_res, failed_outputs

    def refine(self, correct, wrong, question, answer_gt, failed_outputs):
        """Step 2: Refine"""
        if not wrong: return []

        # 1. 讀取與格式化 System Prompt
        sys_tpl = self.meta_prompts.get("analyze_and_rewrite_system", "")
        try:
            sys_msg = sys_tpl.format(num=len(wrong))
        except:
            sys_msg = sys_tpl

        # 2. 準備資料區塊
        error_cases = []
        for p in wrong:
            raw_out = failed_outputs.get(p, "")
            snippet = raw_out[:300] + "..." if len(raw_out) > 300 else raw_out
            error_cases.append(f"<CASE>\nOriginal Prompt: {p}\nModel Output: {snippet}\n</CASE>")
        error_block_str = "\n".join(error_cases)
        
        # 3. 讀取與格式化 User Prompt
        user_tpl = self.meta_prompts.get("analyze_and_rewrite_user", "")
        # 注意: 這裡的 key (question, answer_gt...) 要對應 TXT 裡的 {question}, {answer_gt}
        try:
            user_msg = user_tpl.format(
                question=question, 
                answer_gt=answer_gt, 
                error_block=error_block_str, 
                correct_prompts=correct
            )
        except Exception as e:
            print(f"  [⚠️ Template Error] refine_user: {e}")
            user_msg = f"Question: {question}\nFailed:\n{error_block_str}" # Fallback

        # 4. 發送請求
        response = self.optimizer.chat(sys_msg, user_msg)
        
        improved = text_tools.extract_tags(response, "REWRITE")
        # ... (Log 邏輯保持不變) ...
        if not improved:
            print(f"  [⚠️ WARNING] Refine failed! No tags found.")

        pairs = []
        for i in range(min(len(wrong), len(improved))):
            pairs.append((wrong[i], improved[i]))
        return pairs

    def extract_rule(self, correct, pairs):
        """Step 3: Extract Rule"""
        if not pairs: return ""
        
        # 1. System Prompt
        sys_msg = self.meta_prompts.get("rule_summarization_system", "")
        
        # 2. 準備資料
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        correct_text = "\n".join([f"- {c}" for c in correct])
        
        # 3. User Prompt
        user_tpl = self.meta_prompts.get("rule_summarization_user", "")
        try:
            user_msg = user_tpl.format(
                pair_block=pair_text, 
                correct_block=correct_text
            )
        except Exception as e:
            print(f"  [⚠️ Template Error] rule_summarization_user: {e}")
            user_msg = f"Pairs:\n{pair_text}\nCorrect:\n{correct_text}"

        return self.optimizer.chat(sys_msg, user_msg)

    def combine_rules(self, rules):
        """Step 4: Combine Rules"""
        if not rules: return ""
        
        # 1. System Prompt
        sys_msg = self.meta_prompts.get("combine_rules_system", "")
        
        # 2. User Prompt
        user_tpl = self.meta_prompts.get("combine_rules_user", "")
        
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        
        try:
            user_msg = user_tpl.format(rules_block=block)
        except Exception as e:
             user_msg = f"Rules:\n{block}"

        return self.optimizer.chat(sys_msg, user_msg)

    def _generate_prompts_from_rule(self, rule_text, count):
        """Helper: Generate Prompts"""
        if not rule_text: return []
        
        # 1. System Prompt
        sys_tpl = self.meta_prompts.get("prompt_generation_system", "")
        try:
            sys_msg = sys_tpl.format(num=count)
        except:
            sys_msg = sys_tpl.replace("{num}", str(count))
            
        # 2. User Prompt
        user_tpl = self.meta_prompts.get("prompt_generation_user", "")
        try:
            user_msg = user_tpl.format(rule_text=rule_text, count=count)
        except Exception:
            user_msg = f"Rule:\n{rule_text}\nCount: {count}"
        
        try:
            raw = self.optimizer.chat(sys_msg, user_msg)
            # ... (解析邏輯保持不變) ...
            prompts = []
            for line in raw.split('\n'):
                line = line.strip()
                if len(line) > 10 and not line.lower().startswith("here"):
                    line = line.strip('"').strip("'")
                    if line[0].isdigit():
                        line = line.split('.', 1)[-1].strip()
                        line = line.split(')', 1)[-1].strip()
                    prompts.append(line)
            return prompts[:count]
        except Exception as e:
            print(f"  [⚠️ Warning] Generate prompts failed: {e}")
            return []

    # ... (其他函式 run, _log_optimization_status 保持不變) ...