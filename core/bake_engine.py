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
        
        self.concurrency = config['execution']['concurrency']
        self.max_retries = config['execution']['max_retries']
        self.group_size = config['bake']['group_size']
        
        self.paths = config['paths']

    def evaluate_parallel(self, query: str, answer_gt: str, prompts: List[str], task_type: str):
        correct, wrong = [], []
        detailed_res = {}
        failed_outputs = {}

        def _worker(p):
            full_input = f"{p}\n\n{query}"
            for _ in range(self.max_retries):
                try:
                    raw = self.scorer.chat("You are a helpful assistant.", full_input)
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
        if not wrong: return []

        error_cases = []
        for p in wrong:
            raw_out = failed_outputs.get(p, "")
            snippet = raw_out[:300] + "..." if len(raw_out) > 300 else raw_out
            error_cases.append(f"<CASE>\nOriginal Prompt: {p}\nModel Output: {snippet}\n</CASE>")
        
        error_block = "\n".join(error_cases)
        sys_msg = self.meta_prompts.get("analyze_and_rewrite", "")
        
        user_msg = (
            f"[TASK CONTEXT]\nQuestion: {question}\nGround Truth: {answer_gt}\n\n"
            f"[FAILED PROMPTS & OUTPUTS]\n{error_block}\n\n"
            f"[SUCCESSFUL PROMPTS (REFERENCE)]\n{correct}"
        )

        response = self.optimizer.chat(sys_msg.format(num=len(wrong)), user_msg)
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
        tpl = self.meta_prompts.get("rule_summarization", "")
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        try:
            sys_msg = tpl.format(pairs_block=pair_text)
        except Exception:
            sys_msg = tpl
        user_msg = f"Correct Prompts:\n{correct}"
        return self.optimizer.chat(sys_msg, user_msg)

    def combine_rules(self, rules):
        if not rules: return ""
        tpl = self.meta_prompts.get("combine_rules", "")
        block = "\n\n".join([f"Rule {i+1}:\n{r}" for i, r in enumerate(rules)])
        try:
            sys_msg = tpl.format(rules_block=block)
        except Exception:
            sys_msg = f"{tpl}\n\nRules:\n{block}"
        return self.optimizer.chat(sys_msg, "Please fill the template based on the rules above.")

    def _generate_prompts_from_rule(self, rule_text, count):
        if not rule_text: return []
        gen_tpl = self.meta_prompts.get("prompt_generation", "")
        try:
            sys_msg = gen_tpl.format(rules_block=rule_text, num=count)
        except Exception:
            sys_msg = gen_tpl.replace("{rules_block}", rule_text).replace("{num}", str(count))
            
        user_msg = f"Please generate {count} new prompts based on the above rule now."
        try:
            raw = self.optimizer.chat(sys_msg, user_msg)
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

    def run(self, dataset, initial_prompts):
        """主流程"""
        current_prompts = initial_prompts.copy()
        attr, all_rule = [], []
        
        opt_status_path = self.paths.get('opt_status', "logs/optimization_status.csv")
        trace_log_path = self.paths.get('trace_log', "logs/refinement_trace.jsonl") 
        # [新增]
        prompt_history_path = self.paths.get('prompt_history', "logs/prompt_history.jsonl")
        rule_evolution_path = self.paths.get('rule_evolution', "logs/rule_evolution.jsonl")
        
        # 初始化 Log
        logger.init_files([
            self.paths['detailed_log'], 
            self.paths['rules_log'], 
            opt_status_path,
            trace_log_path,
            prompt_history_path, # New
            rule_evolution_path  # New
        ])

        # 寫入 CSV 表頭
        if not text_tools.file_has_content(opt_status_path):
             with open(opt_status_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "source", "status", "initial_wrong", "verified_success", "note"])

        # [新增] 記錄初始 Prompts
        logger.log_jsonl(prompt_history_path, {
            "event": "initial_load",
            "sample_idx": 0,
            "prompts": current_prompts,
            "count": len(current_prompts)
        })

        for idx, item in enumerate(dataset):
            q, a = item['question'], item['answer']
            t_type = item.get('type', 'general')
            src = item.get('source', 'unknown')
            
            print(f"Processing {idx+1}/{len(dataset)} [{src}]...")
            
            verified_success_count = 0
            
            # 1. First Eval
            Pc, Pi, details, failed_outputs = self.evaluate_parallel(q, a, current_prompts, task_type=t_type)
            print(f"  > Initial: Correct: {len(Pc)}, Wrong: {len(Pi)}")
            
            logger.log_jsonl(self.paths['detailed_log'], {
                "id": idx, "source": src, "type": t_type, 
                "q": q, "res": details
            })
            
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
                logger.log_jsonl(trace_log_path, {
                    "id": idx, "source": src, 
                    "original_prompt": old_p, "candidate_prompt": new_p,
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
                logger.log_rule(self.paths['rules_log'], f"Sample {idx} ({src})", log_content)

            # 5. Merge Logic & Iterative Update
            if len(attr) >= self.group_size:
                merged = self.combine_rules(attr)
                all_rule.append(merged)
                attr.clear()
                logger.log_rule(self.paths['rules_log'], "Tier-1 Merge", merged)
                
                # [新增] 記錄 Tier-1 Rule
                logger.log_jsonl(rule_evolution_path, {
                    "sample_idx": idx,
                    "tier": "Tier-1",
                    "rule_content": merged
                })

                # [迭代功能]
                print(f"\n  ⚡ [Iterative Update] Tier-1 Rule generated! Updating prompts...")
                new_iterative_prompts = self._generate_prompts_from_rule(merged, count=5)
                
                if new_iterative_prompts:
                    current_prompts = new_iterative_prompts
                    print(f"  🔄 Prompt Pool Updated: {len(current_prompts)} new prompts.")
                    
                    # [新增] 記錄 Prompt 更新
                    logger.log_jsonl(prompt_history_path, {
                        "event": "iterative_update",
                        "sample_idx": idx,
                        "prompts": current_prompts,
                        "derived_from_rule_tier": "Tier-1",
                        "count": len(current_prompts)
                    })
                else:
                    print("  ⚠️ Failed to generate new prompts.")

            # Recursive Merge
            while len(all_rule) >= self.group_size:
                chunk = all_rule[:self.group_size]
                merged = self.combine_rules(chunk)
                all_rule = [merged] + all_rule[self.group_size:]
                logger.log_rule(self.paths['rules_log'], "Recursive Merge", merged)
                
                # [新增] 記錄 Recursive Rule
                logger.log_jsonl(rule_evolution_path, {
                    "sample_idx": idx,
                    "tier": "Recursive/Tier-N",
                    "rule_content": merged
                })

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
            # [新增] 記錄 Final Convergence Rule
            logger.log_jsonl(rule_evolution_path, {
                "sample_idx": "FINAL",
                "tier": "Convergence",
                "rule_content": merged
            })
            
        final_rule = all_rule[0] if all_rule else ""
        logger.log_rule(self.paths['rules_log'], "FINAL RULE", final_rule)
        
        final_prompts = self._generate_prompts_from_rule(final_rule, count=self.config['bake']['max_output_prompts'])
        
        return final_prompts, final_rule

    def _log_optimization_status(self, filepath, idx, src, status, initial_wrong, verified_success, note):
        with open(filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, src, status, initial_wrong, verified_success, note])