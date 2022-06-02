import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2Tokenizer, OpenAIGPTTokenizer, BertTokenizer, RobertaTokenizer, XLNetTokenizer, \
    AlbertTokenizer

TOKENIZERS = {
    "gpt2": GPT2Tokenizer,
    "openai-gpt": OpenAIGPTTokenizer,
    "bert-base-uncased": BertTokenizer,
    "bert-base-cased": BertTokenizer,
    "roberta-base": RobertaTokenizer,
    "xlnet-base-cased": XLNetTokenizer,
    "albert-base-v2": AlbertTokenizer,
    "albert-large-v2": AlbertTokenizer,
    "albert-xlarge-v2": AlbertTokenizer
}


def get_model_tokenizer(type_='gpt2'):
    tokenizer = TOKENIZERS[type_].from_pretrained(type_)
    special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def set_seed(cnf):
    random.seed(cnf["seed"])
    np.random.seed(cnf["seed"])
    torch.manual_seed(cnf["seed"])
    if cnf["n_gpu"] > 0:
        torch.cuda.manual_seed_all(cnf["seed"])


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, context, length, device, temperature=1.0, top_k=0.0, top_p=0.0):
    """ Generates a sequence of tokens
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """

    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            outputs = model(
                **inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def generate_summaries(ds, tokenizer, model, length, temperature=0.8, top_k=10, top_p=0.5, device="cuda", verbose=False):
    result = []
    for i in range(len(ds)):
        context, summary = ds[i]['document'], ds[i]['summary']
        generated_text = sample_seq(model, context, len(summary) if length is None else length, device, temperature, top_k, top_p)
        generated_text = generated_text[0, len(context):].tolist()
        text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        result.append(text)

        if verbose:
            print(f"Article:")
            print(tokenizer.decode(context))
            print("=" * 50)
            print(f"Generated summary:")
            print(text)
            print("=" * 50)
            print(f"Actual summary:")
            print(tokenizer.decode(summary))
            print("=" * 50)
    return result
