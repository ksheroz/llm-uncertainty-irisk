"""Semantic clustering via bidirectional NLI entailment (Farquhar et al., 2024)."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

NLI_LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}


class SemanticClusterer:
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def check_entailment(self, premise: str, hypothesis: str) -> str:
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return NLI_LABELS[logits.argmax(dim=-1).item()]

    def are_equivalent(self, s_a: str, s_b: str, context: str = "") -> bool:
        if context:
            s_a = f"Question: {context} Answer: {s_a}"
            s_b = f"Question: {context} Answer: {s_b}"
        return (
            self.check_entailment(s_a, s_b) == "entailment"
            and self.check_entailment(s_b, s_a) == "entailment"
        )

    def cluster(self, generations: List[str], context: str = "") -> List[List[str]]:
        clusters = []
        for gen in generations:
            assigned = False
            for c in clusters:
                if self.are_equivalent(gen, c[0], context):
                    c.append(gen)
                    assigned = True
                    break
            if not assigned:
                clusters.append([gen])
        return clusters
