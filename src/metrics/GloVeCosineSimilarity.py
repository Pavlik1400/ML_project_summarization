import numpy as np

from .nlp_metric import NLPMetric
from ..constants import CORPUS_T
from ..embedings import GloVe
from sklearn.metrics.pairwise import cosine_similarity


class GloveCosineSimilarity(NLPMetric):
    def __init__(self):
        self.embeder = GloVe()
        self.embeder.initalize()
        self.predictions = []
        self.references = []

    def update_state(self, pred: CORPUS_T, actual: CORPUS_T):
        for p in pred:
            self.predictions.append(p)
        for a in actual:
            self.references.append(a)

    def reset_state(self):
        self.predictions = []
        self.references = []

    def result(self) -> float:
        emb_predictions = self.embeder.fit_transform(self.predictions)
        emb_references = self.embeder.fit_transform(self.references)
        return float(np.mean(np.diag(cosine_similarity(emb_predictions, emb_references))))


