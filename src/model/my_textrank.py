import networkx as nx
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .nlp_model import NLPModel, CORPUS_T
from ..embedings import GloVe


class MyTextRankModel(NLPModel):
    def __init__(self, n_sentences=1, embeder=GloVe(), **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = n_sentences
        self.embedder = embeder
        self.embedder.initalize()
        # self.sort_orders = []

    def fit_doc(self, doc: str):
        corpus = np.asarray(nltk.sent_tokenize(doc))
        emb_corpus = self.embedder.fit_transform(corpus)
        n_sentences = len(corpus)
        sim_mat = np.zeros([n_sentences, n_sentences])
        for i in range(n_sentences):
            for j in range(n_sentences):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(emb_corpus[i].reshape(1, 100), emb_corpus[j].reshape(1, 100))[0, 0]
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        sort_order = sorted(scores.keys(), reverse=True, key=lambda i: scores[i])
        return ". ".join(corpus[k] for k in sort_order[:self.n_sentences])
        # return sorted(scores.keys(), reverse=True, key=lambda i: scores[i])

    def fit(self, corpus: CORPUS_T, **kwargs):
        pass

    def predict(self, corpus: CORPUS_T, **kwargs) -> CORPUS_T:
        result = []
        # for idx, doc in enumerate(tqdm(corpus)):
        #     result.append(". ".join([doc[k] for k in self.sort_orders[idx][:self.n_sentences]]))
        for doc in tqdm(corpus):
            result.append(self.fit_doc(doc))

        return result

    def __str__(self):
        return "MyTextRank"
