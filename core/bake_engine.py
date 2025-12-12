import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            full_input = f"{p}\n\n{query}" # MMLU 題目已經包含 "Answer:"
            
            for _ in range(self.max_retries):
                # 呼叫 LLM
                raw = self.scorer.chat("You are a helpful assistant.", full_input)
                
                # [關鍵修改] 使用統一驗證器，傳入 task_type
                if text_tools.validate_answer(raw, answer_gt, task_type):
                    return (p, True)
                
                # 簡單容錯: 如果是連線錯誤才 sleep，如果是答案錯就不 sleep (這裡簡化)
                # time.sleep(1) 
            return (p, False)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_p = {executor.submit(_worker, p): p for p in prompts}
            for future in as_completed(future_to_p):
                p, is_correct = future.result()
                detailed_res[p] = is_correct
                (correct if is_correct else wrong).append(p)
                
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
        
        logger.init_files([self.paths['detailed_log'], self.paths['rules_log']])

        for idx, item in enumerate(dataset):
            # [關鍵] 從資料中讀取 type
            q, a = item['question'], item['answer']
            t_type = item['type'] 
            src = item.get('source', 'unknown')
            
            print(f"Processing {idx+1}/{len(dataset)} [{src}]...")
            
            # 1. Eval (傳入 task_type)
            Pc, Pi, details = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            
            logger.log_jsonl(self.paths['detailed_log'], {
                "id": idx, "source": src, "type": t_type, 
                "q": q, "res": details
            })
            
            if not Pi: continue

            # 2. Refine & Extract
            pairs = self.refine(Pc, Pi)
            rule = self.extract_rule(Pc, pairs)
            
            if rule:
                # 可以在 Rule 前面加上來源標記，幫助 Optimizer 區分
                tagged_rule = f"[Source: {src}] {rule}"
                attr.append(tagged_rule)
                logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", rule)

            # 3. Merge Logic (Recursive)
            if len(attr) >= self.group_size:
                merged = self.combine_rules(attr)
                all_rule.append(merged)
                attr.clear()
                logger.log_rule(self.paths['rules_log'], "Tier-1 Merge", merged)
            
            while len(all_rule) >= self.group_size:
                chunk = all_rule[:self.group_size]
                merged = self.combine_rules(chunk)
                all_rule = [merged] + all_rule[self.group_size:]
                logger.log_rule(self.paths['rules_log'], "Recursive Merge", merged)

        # 4. Finalize
        if attr: all_rule.append(self.combine_rules(attr))
        while len(all_rule) > 1:
            merged = self.combine_rules(all_rule[:self.group_size])
            all_rule = [merged] + all_rule[self.group_size:]
            
        final_rule = all_rule[0] if all_rule else ""
        logger.log_rule(self.paths['rules_log'], "FINAL RULE", final_rule)
        
        # 5. Generate Prompts
        gen_tpl = self.meta_prompts.get("prompt_generation", "")
        sys_msg = gen_tpl.format(rules_block=final_rule, num=self.config['bake']['max_output_prompts'])
        raw = self.optimizer.chat(sys_msg, f"Rule:\n{final_rule}")
        
        return [line.strip() for line in raw.split('\n') if len(line) > 10]