from .nlp_metric import NLPMetric, CORPUS_T
from datasets import load_metric


bleu = load_metric("bleu")


class BLEU(NLPMetric):
    def __init__(self, max_order: int):
        self.predictions = []
        self.references = []
        self.max_order = max_order

    def update_state(self, pred: CORPUS_T, actual: CORPUS_T):
        for p in pred:
            self.predictions.append(p.split())
        for a in actual:
            self.references.append([a.split()])

    def reset_state(self):
        self.predictions = []
        self.references = []

    def result(self) -> float:
        results = bleu.compute(
            predictions=self.predictions,
            references=self.references,
            max_order=self.max_order,
        )

        return results["bleu"]

    def __str__(self):
        return f"BLEU-{self.max_order}"
