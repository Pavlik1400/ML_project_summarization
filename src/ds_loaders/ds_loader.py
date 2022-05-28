from abc import ABC, abstractmethod, abstractproperty
from typing import List, Optional


class DSLoader(ABC):
    def __init__(self, path=""):
        self.X_train: Optional[List] = None
        self.y_train: Optional[List] = None

        self.X_test: Optional[List] = None
        self.y_test: Optional[List] = None

        self.X_val: Optional[List] = None
        self.y_val: Optional[List] = None

        self.path = path

    @abstractmethod
    def load(self):
        raise NotImplemented


class HuggingFaceDSLoader(DSLoader, ABC):
    @abstractproperty
    def article_label(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def summary_label(self) -> str:
        raise NotImplementedError

    def __init__(self, path=""):
        super().__init__(path)

    def HF_ds2splits(self, ds):
        self.X_train = list(ds["train"][self.article_label])
        self.y_train = list(ds["train"][self.summary_label])
        self.X_test = list(ds["test"][self.article_label])
        self.y_test = list(ds["test"][self.summary_label])
        self.X_val = list(ds["validation"][self.article_label])
        self.y_val = list(ds["validation"][self.summary_label])
