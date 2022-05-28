"""
Script to fine tune a model
"""

import json
import time
from argparse import ArgumentParser
from pprint import pformat
from typing import Dict

import torch
import wandb
from pytorch_transformers import GPT2LMHeadModel, AdamW, WarmupLinearSchedule
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tnrange, tqdm

from src.ds_loaders.gpt_2_dataset import GPT21024DatasetTok
from src.utils.deep_tools import set_seed, get_model_tokenizer
from src.utils.logger import LOGGER


# from tensorboardX import SummaryWriter
# from dataset import GPT21024Dataset
# from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering


def init_wandb(name: str, cnf: Dict):
    wandb.init(project="ml_summarization", name=name)
    wb_config = wandb.config
    for k in cnf:
        wb_config.__setattr__(k, cnf[k])
    return wb_config


def train(model, tokenizer, train_dataset, valid_dataset, ignore_index, cnf):
    # """ Trains GPT2 model and logs necessary details.
    # 	Args:
    # 		args: dict that contains all the necessary information passed by user while training
    # 		model: finetuned gpt/gpt2 model
    # 		tokenizer: GPT/GPT2 tokenizer
    # 		train_dataset: GPT21024Dataset object for training data
    # 		ignore_index: token not considered in loss calculation
    # """

    # - Initialize all the stuff
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=cnf["batch_size"],
                          num_workers=cnf["num_workers"])
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(), lr=cnf["lr"])
    scheduler = WarmupLinearSchedule(optimizer, 100, 80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = tnrange(int(cnf["num_train_epochs"]), desc="Epoch")
    if cnf["seed"] is not None:
        set_seed(cnf)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            # get value from dataset
            inputs, labels = torch.tensor(batch['document']), torch.tensor(batch['document'])
            inputs = inputs.to(cnf["device"])
            labels = labels.to(cnf["device"])

            # forward
            model.train()
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item()  # index of separator token

            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx + 1:].contiguous()

            # calculate loss function
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / cnf["gradient_accumulation_steps"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cnf["max_grad_norm"])
            tr_loss += loss.item()

            # if we iterated over gradient_accumulation_steps docs, update weights
            if (step + 1) % cnf["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # wandb.log({"lr": scheduler.get_lr(), "loss": (tr_loss - logging_loss)/cnf["gradient_accumulation_steps"]})
                # logging_loss = tr_loss
                wandb.log({"lr": scheduler.get_lr(), "loss": tr_loss})
                print("loss:", loss.item(), end='\n\n')

                # if (step + 1)/cnf["gradient_accumulation_steps"] == 1.0:
                # 	print('After 1st update: ', end='\n\n')
                # 	generate_sample(valid_dataset, tokenizer, num=2, eval_step=False)

            # if (step + 1) % (10*cnf["gradient_accumulation_steps"]) == 0:
            # results = evaluate(args, model, valid_dataset, ignore_index, global_step)
            # for key, value in results.items():
            #     writer.add_scalar('eval_{}'.format(key), value, global_step)
            # print('After', global_step+1,'updates: ', end='\n\n')
            # generate_sample(valid_dataset, tokenizer, num=2, eval_step=True)


def main(args):
    cnf = json.load(open(args.config, 'r'))
    LOGGER.info(f"config: ")
    LOGGER.into(pformat(cnf))

    LOGGER.debug(f"Init wand")
    wb_config = init_wandb(args.name, cnf)

    LOGGER.debug(f"Create {args.model} tokenizer")
    tokenizer = get_model_tokenizer(args.model)
    ignore_idx = tokenizer.pad_token_id

    LOGGER.debug("Load train data")
    train_data = GPT21024DatasetTok(args.ds_path, mode="train", length=args.train_size)

    LOGGER.debug("Load validation data")
    valid_data = GPT21024DatasetTok(args.ds_path, mode="val", length=args.val_size)

    LOGGER.debug(f"Create {args.model} model")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    LOGGER.debug(f"len of tokenizer: {len(tokenizer)}")

    model.to(cnf["device"])

    LOGGER.info("Start training")
    start = time.time()
    train(args, model, tokenizer, train_data, valid_data, ignore_idx)
    LOGGER.info('total time: ', (time.time() - start) / 60, " minutes", end='\n\n')


if __name__ == '__main__':
    parser = ArgumentParser("script that trains given model with given dataset")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=None, help="clips ds")
    parser.add_argument("--val_size", type=str, default=None, help="clips ds")
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
    main(args)
