from typing import Literal

from datasets import load_metric

from .nlp_metric import NLPMetric
from ..constants import CORPUS_T

rouge_subtypes = [
    "rouge1", "rouge2", "rougeL"
]

rouge_submetrics = {
    "precision": 0,
    "recall": 1,
    "fmeasure": 2,
}

rouge = load_metric("rouge")


class ROUGE(NLPMetric):
    def __init__(self,
                 subtype: Literal["rouge1", "rouge2", "rougeL"],
                 submetric: Literal["precision", "recall", "fmeasure"]):
        self.subtype = subtype
        self.submetric = submetric
        self.predictions = []
        self.references = []

    def update_state(self, pred: CORPUS_T, actual: CORPUS_T):
        for p in pred:
            self.predictions.append(p.split())
        for a in actual:
            self.references.append([a.split()])

    def reset_state(self):
        self.predictions = []
        self.references = []

    def result(self) -> float:
        results = rouge.compute(
            predictions=self.predictions,
            references=self.references,
        )

        # 1 - mid
        return results[self.subtype][1][rouge_submetrics[self.submetric]]

    def __str__(self):
        return f"ROUGE-{self.subtype.replace('rouge', '')}-{self.submetric}"
