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


# core/bake_engine.py

    # [修改 1] 增加回傳 failed_outputs (記錄模型原始的錯誤回答)
    def evaluate_parallel(self, query: str, answer_gt: str, prompts: List[str], task_type: str):
        correct, wrong = [], []
        detailed_res = {}
        failed_outputs = {}  # [NEW] Map: prompt -> raw_output

        def _worker(p):
            full_input = f"{p}\n\n{query}"
            for _ in range(self.max_retries):
                try:
                    # 取得原始回答
                    raw = self.scorer.chat("You are a helpful assistant.", full_input)
                    
                    # 判斷對錯
                    is_correct = text_tools.validate_answer(raw, answer_gt, task_type)
                    
                    # 回傳 (Prompt, IsCorrect, RawOutput)
                    return (p, is_correct, raw)
                    
                except Exception:
                    time.sleep(self.config['execution'].get('retry_delay', 1.0))
            
            return (p, None, None)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_p = {executor.submit(_worker, p): p for p in prompts}
            for future in as_completed(future_to_p):
                p, is_correct, raw_output = future.result()
                
                if is_correct is None: continue # API Error 跳過

                detailed_res[p] = is_correct
                if is_correct:
                    correct.append(p)
                else:
                    wrong.append(p)
                    failed_outputs[p] = raw_output # [NEW] 儲存錯誤內容

        return correct, wrong, detailed_res, failed_outputs

    # [修改 2] 合併 Analyze & Rewrite，並接收 Context
    def refine(self, correct, wrong, question, answer_gt, failed_outputs):
        """
        Step 2: 優化 (One-Pass: Analyze + Rewrite)
        """
        if not wrong: return []

        # 1. 準備 Context (題目、正確答案、失敗的 Prompt 與其對應的錯誤回答)
        error_cases = []
        for p in wrong:
            raw_out = failed_outputs.get(p, "")
            # 截斷過長的輸出以節省 Token
            snippet = raw_out[:300] + "..." if len(raw_out) > 300 else raw_out
            error_cases.append(
                f"<CASE>\nOriginal Prompt: {p}\nModel Output: {snippet}\n</CASE>"
            )
        
        error_block = "\n".join(error_cases)
        
        # 2. 載入新的合併版 Meta-Prompt
        # 假設我們將新 Prompt 存為 'analyze_and_rewrite.txt'
        sys_msg = self.meta_prompts.get("analyze_and_rewrite", "")
        
        # 3. 組合 User Message
        user_msg = (
            f"[TASK CONTEXT]\nQuestion: {question}\nGround Truth: {answer_gt}\n\n"
            f"[FAILED PROMPTS & OUTPUTS]\n{error_block}\n\n"
            f"[SUCCESSFUL PROMPTS (REFERENCE)]\n{correct}"
        )

        # 4. 單次呼叫 Optimizer
        response = self.optimizer.chat(sys_msg.format(num=len(wrong)), user_msg)
        
        # 5. 提取結果
        # 這裡我們只抓取 <REWRITE> 標籤內的內容
        improved = text_tools.extract_tags(response, "REWRITE")
        
        # 配對 (確保數量一致)
        pairs = []
        for i in range(min(len(wrong), len(improved))):
            pairs.append((wrong[i], improved[i]))
            
        return pairs

    def extract_rule(self, correct, pairs):
        """Step 3: 提取規則"""
        if not pairs: return ""
        tpl = self.meta_prompts.get("rule_summarization", "")
        
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        
        # [修正] 嘗試填入模板參數 pairs_block
        try:
            sys_msg = tpl.format(pairs_block=pair_text)
        except Exception:
            sys_msg = tpl
            
        # user_msg 可以簡化，因為內容已經在 system prompt (或者保留以防萬一)
        user_msg = f"Correct Prompts:\n{correct}"
        
        return self.optimizer.chat(sys_msg, user_msg)

    def combine_rules(self, rules):
        """Step 4: 合併規則 (使用結構化模板)"""
        if not rules: return ""
        
        tpl = self.meta_prompts.get("combine_rules", "")
        
        # 將多條規則組合成文字塊
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        
        # [關鍵修正] 確保填入模板參數 rules_block
        try:
            # 這裡將 block 填入 combine_rules.txt 的 {rules_block} 位置
            sys_msg = tpl.format(rules_block=block)
        except Exception:
            # 防呆：如果模板格式有誤，退回簡單串接
            sys_msg = f"{tpl}\n\nRules:\n{block}"
            
        # 因為規則與指令都已經包含在 System Prompt (sys_msg) 裡了
        # User Prompt 只需要給一個簡單的觸發指令
        return self.optimizer.chat(sys_msg, "Please fill the template based on the rules above.")


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
            
            # 1. First Eval (評測原始 Prompt)
            # 這裡回傳 failed_outputs 是為了給 Refine 做診斷用
            Pc, Pi, details, failed_outputs = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            
            print(f"  > Initial: Correct: {len(Pc)}, Wrong: {len(Pi)}")
            
            # Log 詳細結果
            logger.log_jsonl(self.paths['detailed_log'], {
                "id": idx, "source": src, "type": t_type, 
                "q": q, "res": details
            })
            
            if not Pi: continue

            # 2. Refine (產生「候選」的新 Prompt)
            # 這些只是 Optimizer 覺得會比較好，但還沒被證實
            candidate_pairs = self.refine(Pc, Pi, q, a, failed_outputs)
            
            if not candidate_pairs: continue

            # =================================================
            # 3. [NEW] Verification Step (驗證步驟)
            # =================================================
            # 提取出所有新生成的 Prompt
            new_prompts_to_test = [new_p for (old_p, new_p) in candidate_pairs]
            
            print(f"  > Verifying {len(new_prompts_to_test)} candidates...")
            
            # 再次評測 (只針對題目 q 進行驗證)
            # 注意：這裡我們不關心 failed_outputs，只關心 Pc_new (哪些答對了)
            Pc_new, Pi_new, details_new, _ = self.evaluate_parallel(q, a, new_prompts_to_test, task_type=t_type)
            
            print(f"  > Verification Result: {len(Pc_new)} succeeded, {len(Pi_new)} failed.")

            # 4. Filter Pairs (過濾 Pairs)
            # 只有當 new_p 在 Pc_new (驗證成功列表) 中，才保留該 Pair
            valid_pairs = []
            for old_p, new_p in candidate_pairs:
                if new_p in Pc_new: # 關鍵：確認這個新 Prompt 真的能解對這題
                    valid_pairs.append((old_p, new_p))
            
            # 如果驗證後沒有半個成功的，就跳過這題的規則提取
            if not valid_pairs:
                print("  > No improvements verified. Skipping rule extraction.")
                continue

            # =================================================
            
            # 5. Extract Rule (使用驗證過的 Pairs)
            # 這裡傳入 Pc (原始就對的) 和 valid_pairs (修正後變對的)
            rule = self.extract_rule(Pc, valid_pairs)
            
            if rule:
                attr.append(rule)
                
                # Log 記錄：只記錄那些「原本錯、後來改對」的案例，這才是最有價值的
                failed_prompts_text = "\n".join([f"   [Original X] {old}\n   [Fixed O]    {new}" for old, new in valid_pairs])
                log_content = f"Successful Refinements:\n{failed_prompts_text}\n\nGenerated Guideline:\n{rule}"
                
                logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", log_content)

            # ... (後續 Merge 邏輯保持不變)

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