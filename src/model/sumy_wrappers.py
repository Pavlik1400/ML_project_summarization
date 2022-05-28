from tqdm import tqdm
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
# from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from .nlp_model import NLPModel, CORPUS_T


def sumy_wrapper_generator(summarizer):
    class SumyWrapper(NLPModel):
        def __init__(self, n_sentences=1, **kwargs):
            super().__init__(**kwargs)
            self._inner_model = summarizer()
            self.n_sentences = n_sentences
            # self._parser = None

        def predict_doc(self, doc: str):
            parser = PlaintextParser.from_string(doc, Tokenizer('english'))
            summary = self._inner_model(parser.document, sentences_count=self.n_sentences)
            return " ".join(list(map(lambda x: str(x), summary)))

        def predict(self, corpus: CORPUS_T, **kwargs) -> CORPUS_T:
            result = []
            for doc in tqdm(corpus):
                result.append(self.predict_doc(doc))
            return result

        def fit(self, corpus: CORPUS_T, **kwargs):
            pass

        def __str__(self):
            return f"{summarizer.__name__.replace('Summarizer', '')}"

    return SumyWrapper


TextRankModel = sumy_wrapper_generator(TextRankSummarizer)

LexRankModel = sumy_wrapper_generator(LexRankSummarizer)

LsaModel = sumy_wrapper_generator(LsaSummarizer)

LuhnModel = sumy_wrapper_generator(LuhnSummarizer)

KLModel = sumy_wrapper_generator(KLSummarizer)

# probably should be a little special
# EdmundsonModel = sumy_wrapper_generator(EdmundsonSummarizer)
