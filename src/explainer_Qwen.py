# Qwen Explainer Module

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# configuration for the Qwen/Qwen2.5-1.15B=Instruct model 
@dataclass
class QwenExplainerConfig:
    model_name: str =   QWEN_MODEL
    max_new_tokens: int = 160
    temperature: float = 0.5


class QwenExplainer:
    """
    Uses Qwen/Qwen2.5-1.15B=Instruct to generate natural language
    explanations for predicted logical fallacies in climate posts.
    """

    def __init__(self, cfg: Optional[QwenExplainerConfig] = None):
        self.cfg = cfg or QwenExplainerConfig()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, text: str, label_name: str) -> str:
        return f"""You are an expert in argumentation and logical fallacies.

Task:
Given a social media post about climate change and a predicted logical fallacy label, 
explain in 2–3 short sentences:
1) What this fallacy means in simple terms, and
2) Why this text is (or might be) an example of that fallacy.

Post:
\"\"\"{text}\"\"\"

Predicted fallacy: {label_name}

Explanation:"""

    def explain(self, text: str, label_name: str) -> str:
        prompt = self.build_prompt(text, label_name)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generates model output
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip off the prompt if Qwen echoes or repeats it
        if "Explanation:" in full_text:
            explanation = full_text.split("Explanation:", 1)[-1].strip()
        else:
            explanation = full_text.strip()

        return explanation