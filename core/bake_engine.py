import time
import csv
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import text_tools, logger

class BakeEngine:
    # 定義統一的預設 System Prompt
    DEFAULT_SYS_MSG = "You are a helpful assistant."

    def __init__(self, scorer, optimizer, config, meta_prompts):
        self.scorer = scorer
        self.optimizer = optimizer
        self.config = config
        self.meta_prompts = meta_prompts
        
        self.concurrency = config['execution']['concurrency']
        self.max_retries = config['execution']['max_retries']
        self.group_size = config['bake']['group_size']
        
        # 讀取是否啟用迭代 (預設 False)
        self.enable_iterative = config['bake'].get('iterative', False)
        
        self.paths = config['paths']

    def evaluate_parallel(self, query: str, answer_gt: str, prompts: List[str], task_type: str):
        correct, wrong = [], []
        detailed_res = {}
        failed_outputs = {}

        def _worker(p):
            full_input = f"{p}\n\n{query}"
            for _ in range(self.max_retries):
                try:
                    # [修改] 使用統一的預設 System Prompt
                    raw = self.scorer.chat(self.DEFAULT_SYS_MSG, full_input)
                    # 判斷對錯
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
        """Step 2: 優化 (Analyze + Rewrite)"""
        if not wrong: return []

        error_cases = []
        for p in wrong:
            raw_out = failed_outputs.get(p, "")
            snippet = raw_out[:300] + "..." if len(raw_out) > 300 else raw_out
            error_cases.append(
                f"<CASE>\nOriginal Prompt: {p}\nModel Output: {snippet}\n</CASE>"
            )
        
        error_block = "\n".join(error_cases)
        
        # [修改] 取得指令模板
        template_text = self.meta_prompts.get("analyze_and_rewrite", "")
        
        # [修改] 將指令與任務內容合併到 User Message
        full_user_msg = (
            f"{template_text.format(num=len(wrong))}\n"
            f"----------------------------------------\n"
            f"[TASK CONTEXT]\nQuestion: {question}\nGround Truth: {answer_gt}\n\n"
            f"[FAILED PROMPTS & OUTPUTS]\n{error_block}\n\n"
            f"[SUCCESSFUL PROMPTS (REFERENCE)]\n{correct}"
        )

        # [修改] System 使用預設值
        response = self.optimizer.chat(self.DEFAULT_SYS_MSG, full_user_msg)
        improved = text_tools.extract_tags(response, "REWRITE")
        
        if not improved:
            print(f"  [⚠️ WARNING] Refine failed! No tags found.")
            with open("logs/optimizer_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*20} FAILED PARSE {time.strftime('%X')} {'='*20}\n")
                f.write(f"Response:\n{response}\n")

        pairs = []
        for i in range(min(len(wrong), len(improved))):
            pairs.append((wrong[i], improved[i]))
            
        return pairs

    def extract_rule(self, correct, pairs):
        if not pairs: return ""
        
        # 1. 準備資料字串
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        correct_text = "\n".join([f"- {c}" for c in correct])
        
        # 2. 讀取 Template
        tpl = self.meta_prompts.get("rule_summarization", "")
        
        try:
            # [修改] 填入 txt 定義好的兩個變數位置
            full_user_msg = tpl.format(pairs_block=pair_text, correct_block=correct_text)
        except Exception as e:
            print(f"  [⚠️ Template Format Error] {e}")
            # Fallback: 萬一 txt 檔變數名寫錯，做個簡單串接避免 crash
            full_user_msg = f"{tpl}\n\n[Pairs]:\n{pair_text}\n\n[Correct]:\n{correct_text}"
            
        # 3. 發送請求 (System 使用預設值)
        return self.optimizer.chat(self.DEFAULT_SYS_MSG, full_user_msg)

    def combine_rules(self, rules):
        """Step 4: 合併規則"""
        if not rules: return ""
        
        tpl = self.meta_prompts.get("combine_rules", "")
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        
        try:
            instruction_content = tpl.format(rules_block=block)
        except Exception:
            instruction_content = f"{tpl}\n\nRules:\n{block}"
        
        # [修改] 合併指令與請求
        full_user_msg = f"{instruction_content}\n\nPlease fill the template based on the rules above."
            
        return self.optimizer.chat(self.DEFAULT_SYS_MSG, full_user_msg)

    def _generate_prompts_from_rule(self, rule_text, count):
        """[Helper] 根據規則生成 Prompts"""
        if not rule_text: return []
        
        gen_tpl = self.meta_prompts.get("prompt_generation", "")
        try:
            instruction_content = gen_tpl.format(rules_block=rule_text, num=count)
        except Exception:
            instruction_content = gen_tpl.replace("{rules_block}", rule_text).replace("{num}", str(count))
            
        # [修改] 合併指令與請求
        full_user_msg = f"{instruction_content}\n\nPlease generate {count} new prompts based on the above rule now."
        
        try:
            # [修改] System 使用預設值
            raw = self.optimizer.chat(self.DEFAULT_SYS_MSG, full_user_msg)
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

    # [關鍵修正] 確保這裡有這個函式
    def _log_optimization_status(self, filepath, idx, src, status, initial_wrong, verified_success, note):
        with open(filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, src, status, initial_wrong, verified_success, note])


    def run(self, dataset, initial_prompts):
        """主流程"""
        current_prompts = initial_prompts.copy()
        attr, all_rule = [], []
        
        opt_status_path = self.paths.get('opt_status', "logs/optimization_status.csv")
        trace_log_path = self.paths.get('trace_log', "logs/refinement_trace.jsonl") 
        prompt_history_path = self.paths.get('prompt_history', "logs/prompt_history.jsonl")
        rule_evolution_path = self.paths.get('rule_evolution', "logs/rule_evolution.jsonl")
        
        logger.init_files([
            self.paths['detailed_log'], self.paths['rules_log'], 
            opt_status_path, trace_log_path, prompt_history_path, rule_evolution_path
        ])

        if not text_tools.file_has_content(opt_status_path):
             with open(opt_status_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "source", "status", "initial_wrong", "verified_success", "note"])

        # 記錄初始 Prompts
        logger.log_jsonl(prompt_history_path, {
            "event": "initial_load", "sample_idx": 0, "prompts": current_prompts, "count": len(current_prompts)
        })

        # [修改 1] 取得資料總數，用於判斷是否為最後 10 筆
        total_samples = len(dataset)

        for idx, item in enumerate(dataset):
            q, a = item['question'], item['answer']
            t_type = item.get('type', 'general')
            src = item.get('source', 'unknown')
            
            print(f"Processing {idx+1}/{total_samples} [{src}]...")
            
            # [修改 2] 定義是否紀錄詳細 log (只留前 10 個與後 10 個)
            should_log_details = (idx < 10) or (idx >= total_samples - 10)

            # 1. First Eval
            Pc, Pi, details, failed_outputs = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            print(f"  > Initial: Correct: {len(Pc)}, Wrong: {len(Pi)}")
            
            # [修改 3] 套用過濾條件：detailed_results
            if should_log_details:
                logger.log_jsonl(self.paths['detailed_log'], {"id": idx, "source": src, "type": t_type, "q": q, "res": details})
            
            if not Pi:
                self._log_optimization_status(opt_status_path, idx, src, "Skipped (All Correct)", 0, 0, "")
                continue

            # 2. Refine
            candidate_pairs = self.refine(Pc, Pi, q, a, failed_outputs)
            if not candidate_pairs:
                self._log_optimization_status(opt_status_path, idx, src, "Failed (Refine Step)", len(Pi), 0, "No suggestions from optimizer")
                continue

            # 3. Verification
            new_prompts_to_test = [new_p for (old_p, new_p) in candidate_pairs]
            print(f"  > Verifying {len(new_prompts_to_test)} candidates...")
            
            Pc_new, Pi_new, details_new, verify_failed_outputs = self.evaluate_parallel(q, a, new_prompts_to_test, task_type=t_type)
            print(f"  > Verification Result: {len(Pc_new)} succeeded.")

            for old_p, new_p in candidate_pairs:
                is_verified = (new_p in Pc_new)
                raw_out = verify_failed_outputs.get(new_p, "Correct" if is_verified else "No Output")
                
                # [修改 4] 套用過濾條件：refinement_trace
                if should_log_details:
                    logger.log_jsonl(trace_log_path, {
                        "id": idx, "source": src, "original_prompt": old_p, "candidate_prompt": new_p,
                        "verified": is_verified, "model_output": raw_out
                    })

            valid_pairs = [(old, new) for old, new in candidate_pairs if new in Pc_new]
            verified_success_count = len(valid_pairs)

            if not valid_pairs:
                self._log_optimization_status(opt_status_path, idx, src, "Failed (Verification)", len(Pi), 0, "All candidates failed")
                continue
            else:
                self._log_optimization_status(opt_status_path, idx, src, "Success", len(Pi), verified_success_count, "")

            # 4. Extract Rule
            rule = self.extract_rule(Pc, valid_pairs)
            if rule:
                attr.append(rule)
                failed_prompts_text = "\n".join([f"   [Original X] {old}\n   [Fixed O]    {new}" for old, new in valid_pairs])
                log_content = f"Successful Refinements:\n{failed_prompts_text}\n\nGenerated Guideline:\n{rule}"
                
                # [修改 5] 套用過濾條件：rules_history (僅過濾單一樣本紀錄，保留 Merge 紀錄)
                if should_log_details:
                    logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", log_content)

            # 5. Merge Logic
            if len(attr) >= self.group_size:
                merged = self.combine_rules(attr)
                all_rule.append(merged)
                attr.clear()
                # 保留 Merge 紀錄，不進行過濾，以便查看規則演變結構
                logger.log_rule(self.paths['rules_log'], "Tier-1 Merge", merged)
                
                logger.log_jsonl(rule_evolution_path, {"sample_idx": idx, "tier": "Tier-1", "rule_content": merged})

                # [迭代開關]
                if self.enable_iterative:
                    print(f"\n  ⚡ [Iterative Update] Enabled. Updating prompts from Tier-1 Rule...")
                    
                    iter_count = self.config['bake'].get('iterative_prompt_count', 5)
                    new_iterative_prompts = self._generate_prompts_from_rule(merged, count=iter_count)
                    
                    if new_iterative_prompts:
                        current_prompts = new_iterative_prompts
                        print(f"  🔄 Prompt Pool Updated: {len(current_prompts)} new prompts.")
                        logger.log_jsonl(prompt_history_path, {
                            "event": "iterative_update", "sample_idx": idx, "prompts": current_prompts,
                            "derived_from_rule_tier": "Tier-1", "count": len(current_prompts)
                        })
                    else:
                        print("  ⚠️ Failed to generate new prompts, keeping old ones.")
                else:
                    print(f"  ℹ️ [Iterative Update] Disabled. Continuing with existing prompts.")

            # Recursive Merge
            while len(all_rule) >= self.group_size:
                chunk = all_rule[:self.group_size]
                merged = self.combine_rules(chunk)
                all_rule = [merged] + all_rule[self.group_size:]
                logger.log_rule(self.paths['rules_log'], "Recursive Merge", merged)
                logger.log_jsonl(rule_evolution_path, {"sample_idx": idx, "tier": "Recursive/Tier-N", "rule_content": merged})

        # 6. Finalize
        print("\n=== Finalizing Rules ===")
        if attr: 
            tail = self.combine_rules(attr)
            all_rule.append(tail)
            logger.log_rule(self.paths['rules_log'], "Cleanup Tier-0", tail)
            
        while len(all_rule) > 1:
            merged = self.combine_rules(all_rule[:self.group_size])
            all_rule = [merged] + all_rule[self.group_size:]
            logger.log_rule(self.paths['rules_log'], "Convergence Merge", merged)
            logger.log_jsonl(rule_evolution_path, {"sample_idx": "FINAL", "tier": "Convergence", "rule_content": merged})
            
        final_rule = all_rule[0] if all_rule else ""
        logger.log_rule(self.paths['rules_log'], "FINAL RULE", final_rule)
        
        final_prompts = self._generate_prompts_from_rule(final_rule, count=self.config['bake']['max_output_prompts'])
        
        return final_prompts, final_rule