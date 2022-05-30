""
Script to fine tune a model
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
import json
import time
from datetime import datetime
from argparse import ArgumentParser
from pprint import pformat
from typing import Dict

import torch
import wandb
from pytorch_transformers import AdamW, WarmupLinearSchedule#, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange, tqdm

from src.ds_loaders.gpt_2_dataset import ModelDatasetTok, ModelDatasetTokGPU
from src.utils.deep_tools import set_seed, get_model_tokenizer
from src.utils.logger import LOGGER
from src.model.deep_models import MODELS




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
    if not os.path.exists(cnf["model_dir"]):
        os.makedirs(cnf["model_dir"])

    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=cnf["batch_size"],
                          # num_workers=cnf["num_workers"]
                          )
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(), lr=cnf["lr"])
    scheduler = WarmupLinearSchedule(optimizer, 100, 80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    prev_val_loss = 100000

    model.zero_grad()
    train_iterator = trange(int(cnf["num_train_epochs"]), desc="Epoch")
    if cnf["seed"] is not None:
        set_seed(cnf)

    for ep in train_iterator:
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            # get value from dataset
            inputs, labels = torch.tensor(batch['document']), torch.tensor(batch['document'])
            inputs = inputs.to(cnf["device"])
            labels = labels.to(cnf["device"])

            # forward
            model.train()
            logits = model(inputs)[0]
            # print("=" * 50)
            # print(inputs)
            # print(inputs.shape)
            # logits = model(inputs[0])
            # logits = logits[0]
            
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
                wandb.log({"lr": float(scheduler.get_lr()[0]), "loss": (tr_loss - logging_loss)/cnf["gradient_accumulation_steps"]})
                logging_loss = tr_loss
                # wandb.log({"lr": scheduler.get_lr(), "loss": tr_loss})
                # print("loss:", loss.item(), end='\n\n')
                # print(f"loss: {loss.item()}", end="\n\n")

                # if (step + 1)/cnf["gradient_accumulation_steps"] == 1.0:
                # 	print('After 1st update: ', end='\n\n')
                # 	generate_sample(valid_dataset, tokenizer, num=2, eval_step=False)

            if (step) % cnf["validate_each_step"] == 0:
                LOGGER.info("Evaluating")
                results = evaluate(model, valid_dataset, ignore_index, cnf)
                if prev_val_loss > results['val_loss']:
                    torch.save(model.state_dict(), os.path.join(cnf["model_dir"], f"model_best_val.pt"))
                wandb.log(results)
        # model.save(os.path.join(cnf["weights_dir"], f"model_{step}.pt"))
        torch.save(model.state_dict(), os.path.join(cnf["model_dir"], f"model_{ep}.pt"))


def train_with_batches(model, tokenizer, train_dataset, valid_dataset, ignore_index, cnf):
    # """ Trains GPT2 model and logs necessary details.
    # 	Args:
    # 		args: dict that contains all the necessary information passed by user while training
    # 		model: finetuned gpt/gpt2 model
    # 		tokenizer: GPT/GPT2 tokenizer
    # 		train_dataset: GPT21024Dataset object for training data
    # 		ignore_index: token not considered in loss calculation
    # """

    # - Initialize all the stuff
    if not os.path.exists(cnf["model_dir"]):
        os.makedirs(cnf["model_dir"])
    batch_size = cnf["batch_size"]

    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=cnf["batch_size"],
                        #   num_workers=cnf["num_workers"]
                          )
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(), lr=cnf["lr"])
    scheduler = WarmupLinearSchedule(optimizer, 100, 80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    prev_val_loss = 100000

    model.zero_grad()
    train_iterator = trange(int(cnf["num_train_epochs"]), desc="Epoch")
    if cnf["seed"] is not None:
        set_seed(cnf)

    for ep in train_iterator:
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            # get value from dataset
            inputs, labels = torch.tensor(batch['document']), torch.tensor(batch['document'])
            inputs = inputs.to(cnf["device"])
            labels = labels.to(cnf["device"])

            # forward
            model.train()
            output = model(inputs)
            for b in range(batch_size):
                logits = output[b]
                idx = batch['sum_idx'][b].item()  # index of separator token

                # only consider loss on reference summary just like seq2seq models
                shift_logits = logits[..., idx:-1, :].contiguous()
                shift_labels = labels[..., idx + 1:].contiguous()

                # calculate loss function
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / (cnf["gradient_accumulation_steps"] * batch_size)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cnf["max_grad_norm"])
                tr_loss += loss.item()

            # if we iterated over gradient_accumulation_steps docs, update weights
            if (step + 1) % cnf["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                wandb.log({"lr": scheduler.get_lr()[0], "loss": (tr_loss - logging_loss)/cnf["gradient_accumulation_steps"]})
                logging_loss = tr_loss
                # wandb.log({"lr": scheduler.get_lr(), "loss": tr_loss})
                # print("loss:", loss.item(), end='\n\n')
                # print(f"loss: {loss.item()}", end="\n\n")

                # if (step + 1)/cnf["gradient_accumulation_steps"] == 1.0:
                # 	print('After 1st update: ', end='\n\n')
                # 	generate_sample(valid_dataset, tokenizer, num=2, eval_step=False)

            if (step) % cnf["validate_each_step"] == 0:
                LOGGER.info("Evaluating")
                results = evaluate_with_batch(model, valid_dataset, ignore_index, cnf)
                if prev_val_loss > results['val_loss']:
                    torch.save(model.state_dict(), os.path.join(cnf["model_dir"], f"model_best_val.pt"))
                wandb.log(results)
        # model.save(os.path.join(cnf["weights_dir"], f"model_{step}.pt"))
        torch.save(model.state_dict(), os.path.join(cnf["model_dir"], f"model_{ep}.pt"))
    

def evaluate(model, eval_dataset, ignore_index, cnf):
    # """ Returns perplexity score on validation dataset.
    #     Args:
    #         args: dict that contains all the necessary information passed by user while training
    #         model: finetuned gpt/gpt2 model
    #         eval_dataset: GPT21024Dataset object for validation data
    #         global_step: no. of times gradients have backpropagated
    #         ignore_index: token not considered in loss calculation
    # """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=cnf["batch_size"])
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    val_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = batch['document'].to(cnf["device"]), batch['document'].to(cnf["device"])

        with torch.no_grad():
            logits = model(inputs)[0]
            # idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., batch['sum_idx']:-1, :].contiguous()
            shift_labels = labels[..., batch['sum_idx']+1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            val_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    val_loss = val_loss / nb_eval_steps

    return {"val_loss": val_loss}


def evaluate_with_batch(model, eval_dataset, ignore_index, cnf):
    # """ Returns perplexity score on validation dataset.
    #     Args:
    #         args: dict that contains all the necessary information passed by user while training
    #         model: finetuned gpt/gpt2 model
    #         eval_dataset: GPT21024Dataset object for validation data
    #         global_step: no. of times gradients have backpropagated
    #         ignore_index: token not considered in loss calculation
    # """
    batch_size = cnf["batch_size"]
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=cnf["batch_size"])
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    val_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = batch['document'].to(cnf["device"]), batch['document'].to(cnf["device"])

        with torch.no_grad():
            output = model(inputs)
            for b in range(batch_size):
                logits = output[b]
                # idx = batch['sum_idx'].item() # index of separator token
                # only consider loss on reference summary just like seq2seq models
                shift_logits = logits[..., batch['sum_idx']:-1, :].contiguous()
                shift_labels = labels[..., batch['sum_idx']+1:].contiguous()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += lm_loss.mean().item()
        nb_eval_steps += b

    val_loss = val_loss / nb_eval_steps

    return {"val_loss": val_loss}


def main(args):
    cnf = json.load(open(args.config, 'r'))
    cnf["model_dir"] += "_" + args.name
    LOGGER.info(f"config: ")
    LOGGER.info(pformat(cnf))

    LOGGER.debug(f"Init wand")
    wb_config = init_wandb(args.name, cnf)

    LOGGER.debug(f"Create {args.model} tokenizer")
    tokenizer = get_model_tokenizer(args.model)
    ignore_idx = tokenizer.pad_token_id

    LOGGER.debug("Load train data")
    train_data = ModelDatasetTokGPU(
    # train_data = ModelDatasetTok(
        args.ds_path,
        mode="train",
        length=args.train_size,
        token_size=args.token_size,
        model=args.model,
        device=cnf["device"]
        )
    # train_data = ModelDatasetTok(args.ds_path, mode="train", length=args.train_size, token_size=args.token_size)

    LOGGER.debug("Load validation data")
    valid_data = ModelDatasetTokGPU(
    # valid_data = ModelDatasetTok(
        args.ds_path,
        mode="val",
        length=args.val_size,
        token_size=args.token_size,
        model=args.model,
        device=cnf["device"]
        )
    # valid_data = ModelDatasetTok(args.ds_path, mode="val", length=args.val_size, token_size=args.token_size)

    LOGGER.debug(f"Create {args.model} model")
    # model = GPT2LMHeadModel.from_pretrained(args.model)
    model = MODELS[args.model].from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    LOGGER.debug(f"len of tokenizer: {len(tokenizer)}")

    model.to(cnf["device"])

    LOGGER.info("Start training")
    start = time.time()
    train(model, tokenizer, train_data, valid_data, ignore_idx, cnf)
    # train_with_batches(model, tokenizer, train_data, valid_data, ignore_idx, cnf)
    LOGGER.info('total time: ', (time.time() - start) / 60, " minutes", end='\n\n')

    torch.save(model.state_dict(), os.path.join(cnf["model_dir"], f"model_final.pt"))


if __name__ == '__main__':
    parser = ArgumentParser("script that trains given model with given dataset")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=None, help="clips ds")
    parser.add_argument("--val_size", type=int, default=None, help="clips ds")
    parser.add_argument("--token_size", type=int, default=1024)
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
    main(args)
