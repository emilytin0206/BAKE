import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import text_tools, logger

# 引用工具層
from utils import text_tools, logger

class BakeEngine:
    def __init__(self, scorer, optimizer, config, meta_prompts):
        self.scorer = scorer
        self.optimizer = optimizer
        self.config = config
        self.meta_prompts = meta_prompts
        
        # 參數快取
        self.concurrency = config['execution']['concurrency']
        self.max_retries = config['execution']['max_retries']
        self.group_size = config['bake']['group_size']
        
        # Log 路徑
        self.paths = config['paths']


    def evaluate_parallel(self, query: str, answer_gt: str, prompts: List[str], task_type: str):
        """
        Step 1: 併發評測 (增加 task_type 參數)
        """
        correct, wrong = [], []
        detailed_res = {}

        def _worker(p):
            full_input = f"{p}\n\n{query}" 
            
            for _ in range(self.max_retries):
                try:
                    # [修正] 呼叫 API (現在若連線失敗會拋出異常)
                    raw = self.scorer.chat("You are a helpful assistant.", full_input)
                    
                    # [邏輯區分]
                    # 情境 A: API 成功回傳，檢查答案是否正確
                    if text_tools.validate_answer(raw, answer_gt, task_type):
                        return (p, True) # 答對
                    else:
                        # 答錯 (Model Answer Error) -> 直接回傳 False，不需 Retry (除非想測 Consistency)
                        return (p, False)
                        
                except Exception as e:
                    # 情境 B: API 連線錯誤 (API Error) -> 進行 Retry
                    # print(f"API Error retrying: {e}") 
                    time.sleep(self.config['execution'].get('retry_delay', 1.0))
            
            # [修正] 若重試次數耗盡仍是 API 錯誤，回傳 None
            # 表示此 Prompt 因技術問題無法評測，不應被歸類為「Bad Prompt」
            return (p, None)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_p = {executor.submit(_worker, p): p for p in prompts}
            for future in as_completed(future_to_p):
                p, status = future.result()
                
                # [關鍵修正] 過濾掉 API 失敗 (status is None) 的案例
                # 避免將其加入 wrong 列表，導致 Optimizer 試圖優化一個其實沒問題的 Prompt
                if status is None:
                    continue
                
                detailed_res[p] = status
                if status:
                    correct.append(p)
                else:
                    wrong.append(p)
                
        return correct, wrong, detailed_res


    def refine(self, correct, wrong):
        """Step 2: 優化 (Analyze -> Rewrite)"""
        if not wrong: return []
        
        # 1. Analyze
        sys_1 = self.meta_prompts.get("analyze_only", "").format(num=len(wrong))
        user_1 = text_tools.insert_prompts_template(correct, wrong)
        analysis = self.optimizer.chat(sys_1, user_1)
        
        # 2. Rewrite
        sys_2 = self.meta_prompts.get("rewrite_from_analysis", "").format(num=len(wrong))
        rewrites = self.optimizer.chat(sys_2, analysis)
        
        # Extract
        improved = text_tools.extract_tags(rewrites, "REWRITE")
        
        # Align Pairs
        pairs = []
        for i in range(min(len(wrong), len(improved))):
            pairs.append((wrong[i], improved[i]))
        return pairs

    def extract_rule(self, correct, pairs):
        """Step 3: 提取規則"""
        if not pairs: return ""
        tpl = self.meta_prompts.get("rule_summarization", "")
        
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        user_msg = f"Correct:\n{correct}\n\nPairs:\n{pair_text}"
        
        return self.optimizer.chat(tpl, user_msg)

    def combine_rules(self, rules):
        """Step 4: 合併規則"""
        tpl = self.meta_prompts.get("combine_rules", "")
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        return self.optimizer.chat(tpl, f"Combine these:\n{block}")


    def run(self, dataset, initial_prompts):
        """主流程"""
        current_prompts = initial_prompts.copy()
        attr, all_rule = [], []
        
        # 初始化 Log
        logger.init_files([self.paths['detailed_log'], self.paths['rules_log']])

        for idx, item in enumerate(dataset):
            q, a = item['question'], item['answer']
            t_type = item.get('type', 'general')
            src = item.get('source', 'unknown')
            
            print(f"Processing {idx+1}/{len(dataset)} [{src}]...")
            
            # 1. Eval
            Pc, Pi, details = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            
            # [修正 1] 在 Console 顯示詳細對錯數量
            print(f"  > Correct: {len(Pc)}, Wrong: {len(Pi)}")
            
            # Log 詳細結果 (JSONL)
            logger.log_jsonl(self.paths['detailed_log'], {
                "id": idx, "source": src, "type": t_type, 
                "q": q, "res": details
            })
            
            if not Pi: continue

            # 2. Refine & Extract
            pairs = self.refine(Pc, Pi)
            rule = self.extract_rule(Pc, pairs)
            
            if rule:
                attr.append(rule)
                
                # [修正 2] 在 Log 中記錄是哪些 Prompt 錯了，才導致這條 Rule
                # 這樣你回頭看 Log 就知道: "喔，是因為這幾個 Prompt 沒寫清楚，所以才產生這條規則"
                failed_prompts_text = "\n".join([f"   [X] {p}" for p in Pi])
                log_content = f"Failed Prompts:\n{failed_prompts_text}\n\nGenerated Guideline:\n{rule}"
                
                logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", log_content)

            # 3. Merge Logic (Tier-1)
            if len(attr) >= self.group_size:
                merged = self.combine_rules(attr)
                all_rule.append(merged)
                attr.clear()
                logger.log_rule(self.paths['rules_log'], "Tier-1 Merge", merged)
            
            # 4. Recursive Merge
            while len(all_rule) >= self.group_size:
                chunk = all_rule[:self.group_size]
                merged = self.combine_rules(chunk)
                all_rule = [merged] + all_rule[self.group_size:]
                logger.log_rule(self.paths['rules_log'], "Recursive Merge", merged)

        # 5. Finalize
        print("\n=== Finalizing Rules ===")
        if attr: 
            tail = self.combine_rules(attr)
            all_rule.append(tail)
            logger.log_rule(self.paths['rules_log'], "Cleanup Tier-0", tail)
            
        while len(all_rule) > 1:
            merged = self.combine_rules(all_rule[:self.group_size])
            all_rule = [merged] + all_rule[self.group_size:]
            logger.log_rule(self.paths['rules_log'], "Convergence Merge", merged)
            
        final_rule = all_rule[0] if all_rule else ""
        logger.log_rule(self.paths['rules_log'], "FINAL RULE", final_rule)
        
        # 6. Generate Prompts
        gen_tpl = self.meta_prompts.get("prompt_generation", "")
        sys_msg = gen_tpl.format(rules_block=final_rule, num=self.config['bake']['max_output_prompts'])
        raw = self.optimizer.chat(sys_msg, f"Rule:\n{final_rule}")
        
        final_prompts = [line.strip() for line in raw.split('\n') if len(line) > 10]
        
        # [修正 3] 同時回傳 Prompts 和 Rule 字串
        return final_prompts, final_rule