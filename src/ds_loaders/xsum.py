from .ds_loader import HuggingFaceDSLoader
from datasets import load_dataset


class XSumLoader(HuggingFaceDSLoader):
    @property
    def article_label(self) -> str:
        return "document"

    @property
    def summary_label(self) -> str:
        return "summary"

    def __init__(self, path=""):
        super().__init__(path)

    def load(self):
        self.HF_ds2splits(load_dataset("xsum", "default"))