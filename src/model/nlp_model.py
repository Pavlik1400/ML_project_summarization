from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from ..constants import CORPUS_T


class NLPModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, corpus: CORPUS_T, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, corpus: CORPUS_T, **kwargs) -> CORPUS_T:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        return f"NLPModel"
