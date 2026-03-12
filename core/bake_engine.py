import time
import csv
import os
import re  # <--- [新增] 用於解析標籤
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import text_tools, logger

class BakeEngine:
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
        """Step 1: Evaluation"""
        correct, wrong = [], []
        detailed_res = {}
        failed_outputs = {}

        # [讀取模板] System: 角色設定 / User: 拼接 Prompt 與 Query
        sys_tpl = self.meta_prompts.get("evaluate_system", "You are a helpful assistant.")
        user_tpl = self.meta_prompts.get("evaluate_user", "{prompt}\n\n{query}")

        def _worker(p):
            # [組裝] User Message
            try:
                full_input = user_tpl.format(prompt=p, query=query)
            except Exception:
                full_input = f"{p}\n\n{query}" # Fallback

            for _ in range(self.max_retries):
                try:
                    # [呼叫] 傳入 System 與 User
                    raw = self.scorer.chat(sys_tpl, full_input)
                    is_correct = text_tools.validate_answer(raw, answer_gt, task_type, input_text=query)
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
        """Step 2: Refine (Analyze + Rewrite)"""
        if not wrong: return []

        # 1. [System Message] 讀取並格式化 (例如 {num})
        sys_tpl = self.meta_prompts.get("analyze_and_rewrite_system", "")
        try:
            sys_msg = sys_tpl.format(num=len(wrong))
        except:
            sys_msg = sys_tpl

        # 2. 準備錯誤案例區塊
        error_cases = []
        for p in wrong:
            raw_out = failed_outputs.get(p, "")
            snippet = raw_out[:300] + "..." if len(raw_out) > 300 else raw_out
            error_cases.append(f"<CASE>\nOriginal Prompt: {p}\nModel Output: {snippet}\n</CASE>")
        error_block_str = "\n".join(error_cases)
        
        # 3. [User Message] 讀取並注入資料
        user_tpl = self.meta_prompts.get("analyze_and_rewrite_user", "")
        try:
            user_msg = user_tpl.format(
                question=question, 
                answer_gt=answer_gt, 
                error_block=error_block_str, 
                correct_prompts=correct
            )
        except Exception as e:
            print(f"  [⚠️ Template Error] refine_user: {e}")
            # Fallback
            user_msg = f"Question: {question}\nFailed:\n{error_block_str}\nCorrect:\n{correct}"

        # 4. 發送請求
        response = self.optimizer.chat(sys_msg, user_msg)
        
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
        """Step 3: Extract Rule"""
        if not pairs: return ""
        
        sys_msg = self.meta_prompts.get("rule_summarization_system", "")
        
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        correct_text = "\n".join([f"- {c}" for c in correct])
        
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
        
        sys_msg = self.meta_prompts.get("combine_rules_system", "")
        user_tpl = self.meta_prompts.get("combine_rules_user", "")
        
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        
        try:
            user_msg = user_tpl.format(rules_block=block)
        except Exception:
            user_msg = f"Rules:\n{block}"
            
        return self.optimizer.chat(sys_msg, user_msg)

    def _generate_prompts_from_rule(self, rule_text, count):
        """[Helper] 根據規則生成 Prompts (支援 XML <prompt> 格式)"""
        if not rule_text: return []
        
        # 1. System Message
        sys_tpl = self.meta_prompts.get("prompt_generation_system", "")
        try:
            sys_msg = sys_tpl.format(num=count)
        except:
            sys_msg = sys_tpl.replace("{num}", str(count))
            
        # 2. User Message
        user_tpl = self.meta_prompts.get("prompt_generation_user", "")
        try:
            user_msg = user_tpl.format(rule_text=rule_text, count=count)
        except Exception:
            user_msg = f"Rule:\n{rule_text}"
        
        try:
            raw = self.optimizer.chat(sys_msg, user_msg)
            
            # --- [修改] 使用 Regex 解析 <prompt> ---
            prompts = re.findall(r'<prompt>(.*?)</prompt>', raw, re.DOTALL | re.IGNORECASE)
            prompts = [p.strip() for p in prompts if p.strip()]

            # Fallback: 舊邏輯 (如果模型沒輸出標籤)
            if not prompts:
                print("  [⚠️ Warning] No <prompt> tags found, falling back to line parsing.")
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

    def _save_flattened_prompts(self, prompts, filepath):
        """
        [Helper] 將 Prompts 壓平 (換行轉 \\n) 並存檔。
        方便使用者直接拿這個檔案當作下一次實驗的 initial_prompts。
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for p in prompts:
                    # 將實際換行符號轉為字串 "\\n"，保持一行一條
                    flat_p = p.replace('\n', '\\n')
                    f.write(flat_p + "\n")
            print(f"  💾 Saved flattened prompts to: {filepath}")
        except Exception as e:
            print(f"  [⚠️ Error] Failed to save prompts: {e}")

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
        
        # [新增] 用於儲存最新的 Iterative Prompts
        current_iter_prompts_path = os.path.join(os.path.dirname(opt_status_path), "current_iter_prompts.txt")

        logger.init_files([
            self.paths['detailed_log'], self.paths['rules_log'], 
            opt_status_path, trace_log_path, prompt_history_path, rule_evolution_path
        ])

        if not text_tools.file_has_content(opt_status_path):
             with open(opt_status_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "source", "status", "initial_wrong", "verified_success", "note"])

        logger.log_jsonl(prompt_history_path, {
            "event": "initial_load", "sample_idx": 0, "prompts": current_prompts, "count": len(current_prompts)
        })

        total_samples = len(dataset)

        for idx, item in enumerate(dataset):
            q, a = item['question'], item['answer']
            t_type = item.get('type', 'general')
            src = item.get('source', 'unknown')
            
            print(f"Processing {idx+1}/{total_samples} [{src}]...")
            
            should_log_details = (idx < 10) or (idx >= total_samples - 10)

            # 1. First Eval
            Pc, Pi, details, failed_outputs = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            print(f"  > Initial: Correct: {len(Pc)}, Wrong: {len(Pi)}")
            
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
                if should_log_details:
                    log_content = f"Guideline:\n{rule}"
                    logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", log_content)

            # 5. Merge Logic (Iterative Update)
            if len(attr) >= self.group_size:
                merged = self.combine_rules(attr)
                all_rule.append(merged)
                attr.clear()
                logger.log_rule(self.paths['rules_log'], "Tier-1 Merge", merged)
                logger.log_jsonl(rule_evolution_path, {"sample_idx": idx, "tier": "Tier-1", "rule_content": merged})

                if self.enable_iterative:
                    print(f"\n  ⚡ [Iterative Update] Enabled. Updating prompts from Tier-1 Rule...")
                    iter_count = self.config['bake'].get('iterative_prompt_count', 5)
                    
                    # [呼叫修改後的生成函式]
                    new_iterative_prompts = self._generate_prompts_from_rule(merged, count=iter_count)
                    
                    if new_iterative_prompts:
                        current_prompts = new_iterative_prompts
                        print(f"  🔄 Prompt Pool Updated: {len(current_prompts)} new prompts.")
                        
                        # [新增] 將新 Prompt 壓平並存檔，方便作為下一次的 initial prompt
                        self._save_flattened_prompts(current_prompts, current_iter_prompts_path)
                        
                        logger.log_jsonl(prompt_history_path, {
                            "event": "iterative_update", "sample_idx": idx, "prompts": current_prompts,
                            "derived_from_rule_tier": "Tier-1", "count": len(current_prompts)
                        })
                    else:
                        print("  ⚠️ Failed to generate new prompts, keeping old ones.")

            # Recursive Merge (同原邏輯)
            while len(all_rule) >= self.group_size:
                chunk = all_rule[:self.group_size]
                merged = self.combine_rules(chunk)
                all_rule = [merged] + all_rule[self.group_size:]
                logger.log_rule(self.paths['rules_log'], "Recursive Merge", merged)

        # 6. Finalize
        print("\n=== Finalizing Rules ===")
        if attr: 
            tail = self.combine_rules(attr)
            all_rule.append(tail)
            
        while len(all_rule) > 1:
            merged = self.combine_rules(all_rule[:self.group_size])
            all_rule = [merged] + all_rule[self.group_size:]
            
        final_rule = all_rule[0] if all_rule else ""
        logger.log_rule(self.paths['rules_log'], "FINAL RULE", final_rule)
        
        final_prompts = self._generate_prompts_from_rule(final_rule, count=self.config['bake']['max_output_prompts'])
        
        # [新增] 最終結果也存一份 Flattened 版本
        final_prompts_path = os.path.join(os.path.dirname(opt_status_path), "final_optimized_prompts.txt")
        self._save_flattened_prompts(final_prompts, final_prompts_path)
        
        return final_prompts, final_rule