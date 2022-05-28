import re
import string
from typing import Callable, List, Dict, Tuple, Set
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer, WordNetLemmatizer

from .consts import DEFAULT_CONTRACTION_DICT, EMAIL_REGEX


def to_lower() -> Callable:
    def _to_lower(txt: str) -> str:
        return txt.lower()
    return _to_lower


def expand_contractions(contraction_dict=None) -> Callable:
    if contraction_dict is None:
        contraction_dict = DEFAULT_CONTRACTION_DICT
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))

    def _expand_contractions(txt: str) -> str:
        return contraction_re.sub(lambda match: contraction_dict[match.group(0)], txt)

    return _expand_contractions


def remove_symbols(punc=string.punctuation) -> Callable:
    def _remove_symbols(txt: str) -> str:
        return re.sub('[%s]' % re.escape(punc), '', txt)
    return _remove_symbols


def remove_by_regex(regexes: List[str]) -> Callable:
    remove_re = re.compile('(%s)' % '|'.join(regexes))

    def _remove_by_regex(txt: str) -> str:
        return remove_re.sub("", txt)
    return _remove_by_regex


def replace_by_regex(regex2replace: List[Tuple]) -> Callable:
    def _replace_by_regex(txt: str) -> str:
        for reg, rep in regex2replace:
            txt = re.sub(reg, rep, txt)
        return txt
    return _replace_by_regex


def remove_words(words: Set) -> Callable:
    def _remove_words(txt: str) -> str:
        return " ".join([w for w in txt.split() if w not in words])
    return _remove_words


def stem(stemmer=None) -> Callable:
    if stemmer is None:
        stemmer = PorterStemmer()

    def _stem(txt: str) -> str:
        return " ".join([stemmer.stem(w) for w in txt.split()])
    return _stem


def lemmatize(lemmatizer=None) -> Callable:
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    def _lemmatize(txt: str) -> str:
        return " ".join([lemmatizer.lemmatize(w) for w in txt.split()])
    return _lemmatize


def fix_missplelling() -> Callable:
    # maybe TextBlob, autocorrect
    raise NotImplementedError


def normalize() -> Callable:
    raise NotImplementedError


if __name__ == '__main__':
    text = "Hello, I'm Pablo, I ain't shit. I'm going to talon. pavlik@gmail.com; go goes going gone, 123, 123th"
    print(lemmatize()(remove_by_regex([r"\d"])(remove_words({'I', 'shit'})(remove_symbols(string.punctuation)(replace_by_regex([(EMAIL_REGEX, "EMAIL")])(expand_contractions()(text)))))))
