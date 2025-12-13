import time
import csv
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        correct, wrong = [], []
        detailed_res = {}
        failed_outputs = {}

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
                    # 無論對錯，只要是用於 Debug 或 Trace，都可能需要 output，
                    # 但這裡為了節省記憶體，主要存錯誤的 output
                    failed_outputs[p] = raw_output

        return correct, wrong, detailed_res, failed_outputs



    def refine(self, correct, wrong, question, answer_gt, failed_outputs):
        """Step 2: 優化 (Analyze + Rewrite)"""
        if not wrong: return []

        # 1. 準備 Context
        error_cases = []
        for p in wrong:
            raw_out = failed_outputs.get(p, "")
            snippet = raw_out[:300] + "..." if len(raw_out) > 300 else raw_out
            error_cases.append(
                f"<CASE>\nOriginal Prompt: {p}\nModel Output: {snippet}\n</CASE>"
            )
        
        error_block = "\n".join(error_cases)
        sys_msg = self.meta_prompts.get("analyze_and_rewrite", "")
        
        user_msg = (
            f"[TASK CONTEXT]\nQuestion: {question}\nGround Truth: {answer_gt}\n\n"
            f"[FAILED PROMPTS & OUTPUTS]\n{error_block}\n\n"
            f"[SUCCESSFUL PROMPTS (REFERENCE)]\n{correct}"
        )

        # 4. 呼叫 Optimizer
        response = self.optimizer.chat(sys_msg.format(num=len(wrong)), user_msg)
        
        # 5. 提取結果
        improved = text_tools.extract_tags(response, "REWRITE")
        
        # [NEW] 控制台即時警告
        if not improved:
            print(f"  [⚠️ WARNING] Refine failed! No tags found.")
            print(f"  > Optimizer Response Length: {len(response)} chars")
            print(f"  > Head (first 200 chars): {response[:200]!r}...")
            print(f"  > Tail (last 200 chars): ...{response[-200:]!r}")
            # 將完整錯誤記錄到 debug 檔
            with open("logs/optimizer_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*20} FAILED PARSE {time.strftime('%X')} {'='*20}\n")
                f.write(f"Response:\n{response}\n")

        pairs = []
        for i in range(min(len(wrong), len(improved))):
            pairs.append((wrong[i], improved[i]))
            
        return pairs


    def extract_rule(self, correct, pairs):
        """Step 3: 提取規則"""
        if not pairs: return ""
        tpl = self.meta_prompts.get("rule_summarization", "")
        
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        
        try:
            sys_msg = tpl.format(pairs_block=pair_text)
        except Exception:
            sys_msg = tpl
            
        user_msg = f"Correct Prompts:\n{correct}"
        return self.optimizer.chat(sys_msg, user_msg)

    def combine_rules(self, rules):
        """Step 4: 合併規則"""
        if not rules: return ""
        
        tpl = self.meta_prompts.get("combine_rules", "")
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        
        try:
            sys_msg = tpl.format(rules_block=block)
        except Exception:
            sys_msg = f"{tpl}\n\nRules:\n{block}"
            
        return self.optimizer.chat(sys_msg, "Please fill the template based on the rules above.")


    def run(self, dataset, initial_prompts):
        """主流程"""
        current_prompts = initial_prompts.copy()
        attr, all_rule = [], []
        
        # [修改] 從 self.paths 讀取路徑，支援外部動態傳入
        opt_status_path = self.paths.get('opt_status', "logs/optimization_status.csv")
        trace_log_path = self.paths.get('trace_log', "logs/refinement_trace.jsonl") 
        
        # 初始化 Log (確保傳入的是完整的路徑列表)
        logger.init_files([
            self.paths['detailed_log'], 
            self.paths['rules_log'], 
            opt_status_path,
            trace_log_path
        ])

        # 寫入 CSV 表頭
        with open(opt_status_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "source", "status", "initial_wrong", "verified_success", "note"])

        for idx, item in enumerate(dataset):
            q, a = item['question'], item['answer']
            t_type = item.get('type', 'general')
            src = item.get('source', 'unknown')
            
            print(f"Processing {idx+1}/{len(dataset)} [{src}]...")
            
            # --- 狀態變數 ---
            status = "Processing"
            verified_success_count = 0
            
            # 1. First Eval
            Pc, Pi, details, failed_outputs = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            
            print(f"  > Initial: Correct: {len(Pc)}, Wrong: {len(Pi)}")
            
            # Log 詳細結果
            logger.log_jsonl(self.paths['detailed_log'], {
                "id": idx, "source": src, "type": t_type, 
                "q": q, "res": details
            })
            
            if not Pi:
                self._log_optimization_status(opt_status_path, idx, src, "Skipped (All Correct)", 0, 0, "")
                continue

            # 2. Refine (產生候選 Prompt)
            candidate_pairs = self.refine(Pc, Pi, q, a, failed_outputs)
            
            if not candidate_pairs:
                self._log_optimization_status(opt_status_path, idx, src, "Failed (Refine Step)", len(Pi), 0, "No suggestions from optimizer")
                continue

            # =================================================
            # 3. Verification Step (驗證步驟)
            # =================================================
            new_prompts_to_test = [new_p for (old_p, new_p) in candidate_pairs]
            
            print(f"  > Verifying {len(new_prompts_to_test)} candidates...")
            
            # [修改] 這裡同時接收 failed_outputs (verify_failed_outputs)
            Pc_new, Pi_new, details_new, verify_failed_outputs = self.evaluate_parallel(q, a, new_prompts_to_test, task_type=t_type)
            
            print(f"  > Verification Result: {len(Pc_new)} succeeded, {len(Pi_new)} failed.")

            # [NEW] 記錄每一個優化 Prompt 的詳細結果 (Trace Log)
            # 這樣您就能看到「優化後的 Prompt」長什麼樣子，以及它為什麼失敗
            for old_p, new_p in candidate_pairs:
                is_verified = (new_p in Pc_new)
                # 如果驗證失敗，抓取它的輸出 output
                raw_out = verify_failed_outputs.get(new_p, "Correct" if is_verified else "No Output")
                
                logger.log_jsonl(trace_log_path, {
                    "id": idx,
                    "source": src,
                    "original_prompt": old_p,
                    "candidate_prompt": new_p, # 這就是優化後的 Prompt
                    "verified": is_verified,
                    "model_output": raw_out    # 這是該 Prompt 跑出來的結果
                })

            # 4. Filter Pairs (只保留驗證成功的)
            valid_pairs = []
            for old_p, new_p in candidate_pairs:
                if new_p in Pc_new:
                    valid_pairs.append((old_p, new_p))
            
            verified_success_count = len(valid_pairs)

            if not valid_pairs:
                print("  > No improvements verified. Skipping rule extraction.")
                self._log_optimization_status(opt_status_path, idx, src, "Failed (Verification)", len(Pi), 0, "All candidates failed")
                continue
            else:
                self._log_optimization_status(opt_status_path, idx, src, "Success", len(Pi), verified_success_count, "")

            # =================================================
            
            # 5. Extract Rule
            rule = self.extract_rule(Pc, valid_pairs)
            if rule:
                attr.append(rule)
                failed_prompts_text = "\n".join([f"   [Original X] {old}\n   [Fixed O]    {new}" for old, new in valid_pairs])
                log_content = f"Successful Refinements:\n{failed_prompts_text}\n\nGenerated Guideline:\n{rule}"
                logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", log_content)

            # Merge Logic
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
        if not final_rule:
             print("⚠️ No final rule generated. Returning empty list.")
             return [], ""

        gen_tpl = self.meta_prompts.get("prompt_generation", "")
        sys_msg = gen_tpl.format(rules_block=final_rule, num=self.config['bake']['max_output_prompts'])
        raw = self.optimizer.chat(sys_msg, f"Rule:\n{final_rule}")
        
        final_prompts = [line.strip() for line in raw.split('\n') if len(line) > 10]
        
        return final_prompts, final_rule

    def _log_optimization_status(self, filepath, idx, src, status, initial_wrong, verified_success, note):
        with open(filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, src, status, initial_wrong, verified_success, note])