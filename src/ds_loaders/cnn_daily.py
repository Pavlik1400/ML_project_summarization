from datasets import load_dataset

from .ds_loader import HuggingFaceDSLoader


class CNNDailyLoader(HuggingFaceDSLoader):
    @property
    def article_label(self) -> str:
        return "article"

    @property
    def summary_label(self) -> str:
        return "highlights"

    def __init__(self, path: str=""):
        super().__init__(path)

    def load(self):
        # self.HF_ds2splits(load_dataset("cnn_dailymail", "3.0.0"))
        self.HF_ds2splits(load_dataset("ccdv/cnn_dailymail", "3.0.0"))
