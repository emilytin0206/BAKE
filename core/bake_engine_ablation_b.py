from .bake_engine import BakeEngine

class ConciseBakeEngine(BakeEngine):
    """
    Ablation B: Concise Rule Mode
    Forces the optimizer to generate extremely concise, high-level rules 
    to prevent overfitting to specific structures or keywords.
    """

    def extract_rule(self, correct, pairs):
        """
        Override: 使用 'concise' 版本的 prompt 進行規則歸納
        """
        if not pairs: return ""
        
        # 1. 讀取專用的精簡版 System Prompt
        sys_msg = self.meta_prompts.get("rule_summarization_concise_system", "")
        
        # 2. 準備資料 (保持原樣)
        pair_text = "\n".join([f"Original: {o}\nImproved: {n}" for o, n in pairs])
        correct_text = "\n".join([f"- {c}" for c in correct])
        
        # 3. 讀取 User Prompt (可以用原本的，或是建立新的，這裡我們沿用 summarization_user 結構即可)
        user_tpl = self.meta_prompts.get("rule_summarization_user", "")
        
        try:
            user_msg = user_tpl.format(
                pair_block=pair_text, 
                correct_block=correct_text
            )
        except Exception as e:
            print(f"  [⚠️ Template Error] rule_summarization_user: {e}")
            user_msg = f"Pairs:\n{pair_text}\nCorrect:\n{correct_text}"
            
        # 4. 呼叫模型
        return self.optimizer.chat(sys_msg, user_msg)