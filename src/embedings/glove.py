import numpy as np
import os

GLOVE_50D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "glove", "glove.6B.50d.txt")
GLOVE_100D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "glove", "glove.6B.100d.txt")
GLOVE_200D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "glove", "glove.6B.200d.txt")
GLOVE_300D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "glove", "glove.6B.300d.txt")


class GloVe:
    # GloveTypes = {
    #     "6b50d": GLOVE_50D_PATH,
    #     "6b100d": GLOVE_100D_PATH,
    #     "6b200d": GLOVE_200D_PATH,
    #     "6b300d": GLOVE_300D_PATH,
    # }

    def __init__(self, path: str = GLOVE_100D_PATH):
        self.word_embeddings = None
        self.path = path
        self.initalize()

    def fit_transform(self, corpus):
        sentence_vectors = []
        for i in corpus:
            if len(i) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)
        return sentence_vectors

    def clear_mem(self):
        self.word_embeddings = None

    def initalize(self):
        self.word_embeddings = {}
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.word_embeddings[word] = coefs
        print(f"DEBUG: Loaded GloVe embedings from {self.path}")
