import json
import os
import pickle
import sys
import string
import re
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from src.metrics import BLEU, ROUGE
from src.model.deep_models import MODELS
from src.utils.deep_tools import generate_summaries, get_model_tokenizer
from src.preprocessing.corpus_preprocessor import CorpusPreprocessor
from argparse import ArgumentParser
from pytorch_transformers import GPT2LMHeadModel as GPT2LMHeadModelPT


def evaluate_metrics(sum_pred, sum_true):
    def to_lower():
        def _to_lower(txt: str) -> str:
            return txt.lower()

        return _to_lower

    def remove_symbols(punc=string.punctuation):
        def _remove_symbols(txt: str) -> str:
            return re.sub('[%s]' % re.escape(punc), '', txt)

        return _remove_symbols

    preprocessor = CorpusPreprocessor().add(to_lower()).add(remove_symbols())

    bleu1 = BLEU(1)
    bleu2 = BLEU(2)
    bleu3 = BLEU(3)
    bleu4 = BLEU(4)

    rouge1_f1 = ROUGE("rouge1", "fmeasure")
    rouge2_f1 = ROUGE("rouge2", "fmeasure")
    rougeL_f1 = ROUGE("rougeL", "fmeasure")
    rouge1_r = ROUGE("rouge1", "recall")
    rouge2_r = ROUGE("rouge2", "recall")
    rougeL_r = ROUGE("rougeL", "recall")

    metrics = [bleu1, bleu2, bleu3, bleu4,
               rouge1_f1, rouge2_f1, rougeL_f1, rouge1_r, rouge2_r, rougeL_r]

    result = {}
    for m_idx, metric in enumerate(metrics):
        print(f"Calculating {metric}... ", end="")
        metric.reset_state()
        # for s_pred, s_true in zip(sum_pred, sum_true):
        s_pred, s_true = sum_pred, sum_true
        s_pred, s_true = preprocessor.transform(s_pred), preprocessor.transform(s_true)
        metric.update_state(s_pred, s_true)
        value = metric.result()
        result[str(metric)] = value
        print(f"{value} - {m_idx + 1} / {len(metrics)}")

    return result


def main(args):
    tokenizer = get_model_tokenizer(args.model)

    # GPT2LMHeadModelPT
    model = MODELS[args.model].from_pretrained(args.model)
    # model = GPT2LMHeadModelPT.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = WarmupLinearSchedule(optimizer, 100, 80000)
    model.train()
    model.zero_grad()
    model.eval()
    model.load_state_dict(torch.load(args.model_path))

    if args.ds_path.endswith("json"):
        with open(args.ds_path, 'r') as f:
            ds = json.load(f)['test'][:args.size]
    else:
        with open(args.ds_path, 'rb') as f:
            ds = pickle.load(f)['test'][:args.size]

    pred_summaries = generate_summaries(ds, tokenizer, model,
                                        length=args.length, temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                                        device=args.device, verbose=args.verbose)
    true_summaries = [tokenizer.decode(entry['summary']) for entry in ds]

    metrics = evaluate_metrics(pred_summaries, true_summaries)

    with open(args.save_path, 'w') as m_f:
        json.dump(metrics, m_f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser("Script that calculates metrics on test dataset")
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--token_size", type=int, default=1024)
    parser.add_argument("--model", type=str, help=f"{list(MODELS.keys())}", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--size", type=int, default=None, help="clip test ds")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--length", type=int, help="length of summary. If None, then takes length of true summary "
                                                   "(kinda cheating)")
    parser.add_argument("--verbose", action="store_const", const=True, default=False)

    args = parser.parse_args()
    main(args)
