#google/flan-t5-small
 
from dataclasses import dataclass
from typing import Optional

import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


FLAN_MODEL = "google/flan-t5-small"

@dataclass
class FlanExplainerConfig:
    model_name: str = FLAN_MODEL
    max_new_tokens: int = 120
    temperature: float = 0.3


class FlanExplainer:
     """
    Lightweight explainer using FLAN-T5-small.
    Generates short explanations for predicted fallacies.
    """
     def __init__(self, cfg: Optional[FlanExplainerConfig] = None):
        self.cfg = cfg or FlanExplainerConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.cfg.model_name,
        ).to(self.device)

     def build_prompt(self, text: str, label_name: str) -> str:
        return (
            "You are an expert in logical fallacies.\n\n"
            f"Fallacy: {label_name}\n\n"
            "Task: Read the climate-related post below and explain in 2–3 short sentences:\n"
            "1) What this fallacy means in simple terms, and\n"
            "2) Why the post is (or might be) an example of this fallacy.\n\n"
            "IMPORTANT: Do NOT repeat the post. Only give the explanation.\n\n"
            f"Post: {text}\n\n"
            "Explanation:"
        )

     def explain(self, text: str, label_name: str) -> str:
        prompt = self.build_prompt(text, label_name)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
            temperature=self.cfg.temperature,
            num_beams = 4,
            no_repeat_ngram_size = 3
        )

        raw_out = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        # Control: IF FLAN echoes part of the prompt, keep only after "Explanation:"
        if "Explanation:" in raw_out:
            explanation = raw_out.split("Explanation:", 1)[-1].strip()
        else:
            explanation = raw_out

      
        # Final cleanup: keep it reasonably short (e.g. first 3 sentences)
        sentences = re.split(r'(?<=[\.\?\!])\s+', explanation)
        sentences = [s.strip() for s in sentences if s.strip()]
        short_explanation = " ".join(sentences[:3])

        return short_explanation