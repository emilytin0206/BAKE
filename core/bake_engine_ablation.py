from .bake_engine import BakeEngine

class SuccessOnlyBakeEngine(BakeEngine):
    """Ablation Study: Success-Only"""

    def extract_rule(self, correct, pairs):
        if not correct and not pairs: return ""
        
        # 1. System Prompt
        sys_msg = self.meta_prompts.get("rule_summarization_success_only_system", "")

        # 2. 準備資料
        refined_success = [new_p for (old_p, new_p) in pairs]
        all_success = list(set(correct + refined_success))
        prompts_block_str = "\n".join([f"- {p}" for p in all_success])
        
        # 3. User Prompt
        user_tpl = self.meta_prompts.get("rule_summarization_success_only_user", "")
        
        try:
            user_msg = user_tpl.format(prompts_block=prompts_block_str)
        except Exception as e:
            print(f"  [⚠️ Template Error] rule_summarization_success_only_user: {e}")
            user_msg = f"Prompts:\n{prompts_block_str}"
        
        return self.optimizer.chat(sys_msg, user_msg)