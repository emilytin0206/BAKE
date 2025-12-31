# core/bake_engine_ablation.py
from .bake_engine import BakeEngine

class SuccessOnlyBakeEngine(BakeEngine):
    """
    [Ablation Study C] Success-Only BakeEngine
    
    這個 Engine 繼承自原本的 BakeEngine，但覆寫了 `extract_rule` 方法。
    
    目的：
        驗證「從錯誤修正到正確的對比過程 (Contrastive Learning)」是否為關鍵。
        
    作法：
        在提取規則時，隱藏「錯誤 Prompt -> 修正 Prompt」的對照過程。
        只提供「最終成功的 Prompts」列表給 Optimizer 進行歸納。
        
        這樣 Optimizer 就只能看到 "什麼是好的"，而看不到 "如何從壞的變好的"。
    """

    def extract_rule(self, correct, pairs):
        """
        覆寫原本的規則提取邏輯。
        
        Args:
            correct (List[str]): 一開始就答對的 Prompts (Pc)
            pairs (List[Tuple[str, str]]): 修正成功的配對 [(old_prompt, new_prompt), ...]
        """
        # 如果沒有任何成功的 prompt，直接回傳空字串
        if not correct and not pairs:
            return ""

        print("\n  🧪 [Ablation Mode] Running Success-Only Rule Extraction...")
        
        # 1. 提取修正後成功的 Prompts (只取 pair 中的第二個元素 new_p)
        refined_success = [new_p for (old_p, new_p) in pairs]
        
        # 2. 合併「原本就對」與「修正後對」的 Prompts
        #    這樣確保實驗組 (Ablation) 與對照組 (Full BAKE) 看到的「好 Prompt」集合是一樣的，
        #    唯一的差別在於「資訊呈現方式」(List vs Contrastive Pairs)。
        all_success = correct + refined_success
        
        # 3. 去重 (Deduplicate)
        all_success = list(set(all_success))
        
        # 4. 準備 Prompt 資料塊
        prompts_block = "\n".join([f"- {p}" for p in all_success])
        
        # 5. 定義輸出模板 (與主引擎保持一致，加入思考區塊)
        rule_template = (
            "<THOUGHT>\n"
            "(Analyze the common patterns among these successful prompts. "
            "What specific instructions or structures make them effective?)\n"
            "</THOUGHT>\n"
            "<RULE>\n"
            "(Put your concise guideline here)\n"
            "</RULE>"
        )

        # 6. 建構完整的 User Prompt
        #    包含角色設定、資料展示、指令與格式要求
        full_user_msg = (
            "You are an expert in analyzing prompt engineering techniques.\n"
            "Please summarize the common characteristics, logic, and structure of the following successful prompts.\n\n"
            f"[SUCCESSFUL PROMPTS]\n{prompts_block}\n\n"
            "----------------------------------------\n"
            "[FINAL INSTRUCTION]\n"
            "Provide a general guideline based on the prompts above.\n"
            "Focus ONLY on the patterns found in these high-performing prompts.\n"
            "Summarize the attributes/rules into a single concise guideline.\n\n"
            "Please output your answer strictly using this template:\n"
            f"{rule_template}"
        )
        
        # 7. 呼叫 Optimizer (System Message 使用父類別定義的 DEFAULT_SYS_MSG)
        return self.optimizer.chat(self.DEFAULT_SYS_MSG, full_user_msg)