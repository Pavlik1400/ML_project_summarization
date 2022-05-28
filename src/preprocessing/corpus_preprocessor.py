from typing import Callable, Iterable
from tqdm import trange
import time


class CorpusPreprocessor:
    def __init__(self, name="preprocessor", verbose=False):
        self.prep_handlers = []
        self.verbose = verbose
        self.name = name

    def add(self, prep_handler: Callable) -> "CorpusPreprocessor":
        self.prep_handlers.append(prep_handler)
        return self

    def __transform(self, text: str, _str_call=True):
        for idx, handler in enumerate(self.prep_handlers):
            if self.verbose and _str_call:
                print(f"{idx + 1}/{len(self.prep_handlers)}: Performing {handler.__name__}...")
                st = time.time()
            text = handler(text)
            if self.verbose and _str_call:
                print(f"Done in {'%.3f' % (time.time() - st)}")
        return text

    def transform(self, text):
        # corpus
        if isinstance(text, list) and isinstance(text[0], str):
            transformed_text = []
            rangef = trange if self.verbose else range
            for i in rangef(len(text)):
                transformed_text.append(self.__transform(text[i], False))
            return transformed_text
        # string
        elif isinstance(text, str):
            return self.__transform(text)
        else:
            raise ValueError("Text shoule be 'str' or 'List[str]'")

    def reset(self) -> "CorpusPreprocessor":
        self.prep_handlers.clear()
        return self

    def __str__(self):
        return f"CourpusPreprocessor({self.name})"
