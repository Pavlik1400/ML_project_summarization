import random

import numpy as np
import torch
from transformers import GPT2Tokenizer


def get_gpt2_tokenizer(type_='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(type_)
    special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def set_seed(cnf):
    random.seed(cnf["seed"])
    np.random.seed(cnf["seed"])
    torch.manual_seed(cnf["seed"])
    if cnf["n_gpu"] > 0:
        torch.cuda.manual_seed_all(cnf["seed"])