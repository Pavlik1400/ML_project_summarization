"""
Just a most useful copy-past from .ipynb file to generate json dataset on the gcloud
"""

import os
import sys

from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from argparse import ArgumentParser
from src.ds_loaders.cnn_daily import CNNDailyLoader
from src.ds_loaders.xsum import XSumLoader
from src.ds_loaders.ds_loader import DSLoader
from src.utils.deep_tools import get_model_tokenizer
import json
from transformers import GPT2Tokenizer
import pickle


def model_tokenize_save_less_n(loader: DSLoader, path: str, n: int, tokenizer, clip_idx=-1):

    def _sub_tokenize(X, y, msg=""):
        print(msg)
        res = []
        k = 0
        tq = tqdm(total=clip_idx if clip_idx > 0 else len(y))
        for idx, (doc, summ) in enumerate(zip(X, y)):

            if 0 < clip_idx <= k:
                return res

            doc_tok = tokenizer.encode(doc)
            summ_tok = tokenizer.encode(summ)
            if len(doc_tok) + len(summ_tok) + 1 > n:
                continue
            res.append(
                {
                    'document': doc_tok,
                    'summary': summ_tok,
                }
            )
            k += 1
            tq.update()
        return res

    res = {
        'train': _sub_tokenize(loader.X_train, loader.y_train, msg="Train:"),
        'val': _sub_tokenize(loader.X_val, loader.y_val, msg="Validation:"),
        'test': _sub_tokenize(loader.X_test, loader.y_test, msg="Test:"),
    }

    print(f"Number of train entries: {len(res['train'])}")
    print(f"Number of val entries: {len(res['val'])}")
    print(f"Number of test entries: {len(res['test'])}")

    if path.endswith("json"):
        with open(path, 'w') as f:
            json.dump(res, f, indent=4)
    else:
        with open(path, 'wb') as f:
            pickle.dump(res, f)


def main(args):
    if args.type == "cnn_daily":
        print("ds_loader=CNNDailyLoader()")
        ds_loader = CNNDailyLoader()
    elif args.type == "xsum":
        print("ds_loader = XSumLoader()")
        ds_loader = XSumLoader()
    else:
        raise ValueError(f"Incorrect ds type: {args.type}")
    ds_loader.load()

    print(f"Loading {args.model} tokenizer...")
    tokenizer = get_model_tokenizer(args.model)

    model_tokenize_save_less_n(
        ds_loader,
        path=args.save_path,
        n=args.n,
        tokenizer=tokenizer,
        clip_idx=args.size,
    )


if __name__ == '__main__':
    parser = ArgumentParser("hugging face dataset, convert to json, and remove those, those len(embedings) > n")
    parser.add_argument("-n", type=int, default=1024)
    parser.add_argument("--type", type=str, default="cnn_daily", help="cnn_daily/xsum")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Make sure hugging face has tokenizer for it and pytorch_transformers a model")
    parser.add_argument("--size", type=int, default=-1, help="Size of dataset (clips it)")
    parser.add_argument("--save_path", type=str, help="Where to save ds. Should has extension .json")
    args = parser.parse_args()
    main(args)
