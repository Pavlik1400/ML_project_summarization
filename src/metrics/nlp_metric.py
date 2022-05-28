from abc import ABC, abstractmethod

from ..constants import CORPUS_T


class NLPMetric(ABC):
    @abstractmethod
    def update_state(self, pred: CORPUS_T, actual: CORPUS_T):
        raise NotImplementedError

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError

    @abstractmethod
    def result(self) -> float:
        raise NotImplementedError
