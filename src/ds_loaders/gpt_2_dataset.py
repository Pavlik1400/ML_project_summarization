import json
# from typing import Literal

import torch
from torch.utils.data import Dataset

from ..utils.deep_tools import get_model_tokenizer
import pickle


class ModelDatasetTokGPU(Dataset):
    def __init__(self,
                 path: str,
                 #  mode=Literal['train', 'test', 'val'],
                 mode: str,  # 'train', 'test', 'val'
                 length=None,
                 token_size=1024,
                 device="gpu",
                 model="gpt2"):
        self.tokenizer = get_model_tokenizer(model)
        self.token_size = token_size
        self.device = device

        print(f"Loading ds...")
        if path.endswith("json"):
            with open(path, 'r') as ds_f:
                ds = json.load(ds_f)
        else:
            with open(path, 'rb') as ds_f:
                ds = pickle.load(ds_f)
        if mode == 'train':
            ds = ds['train'][:length]
        elif mode == 'val':
            ds = ds['val'][:length]
        elif mode == 'test':
            ds = ds['test'][:length]
        else:
            raise ValueError(f"Incorrect mode: {mode}")

        self.mode = mode
        if length is None:
            self.len = len(ds)
        else:
            self.len = length

        print(f"Moving ds to {self.device}")
        self.ds = []
        self.sum_indecies = []
        for idx in range(self.len):
            text = self.tokenizer.encode(self.tokenizer.pad_token) * self.token_size
            content = ds[idx]['document'] + self.tokenizer.encode(self.tokenizer.sep_token) + ds[idx]['summary']
            text[:len(content)] = content
            text = torch.tensor(text)
            self.ds.append(text.to(self.device))
            self.sum_indecies.append(len(ds[idx]['document']))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'document': self.ds[idx], 'sum_idx': self.sum_indecies[idx]}
        return sample


class ModelDatasetTok(Dataset):
    def __init__(self,
                 path: str,
                #  mode=Literal['train', 'test', 'val'],
                 mode: str,                     # 'train', 'test', 'val'
                 length=None,
                 token_size=1024,
                 model="gpt2"):
        self.tokenizer = get_model_tokenizer(model)
        self.token_size = token_size
        print(f"Loading ds...")
        if path.endswith("json"):
            with open(path, 'r') as ds_f:
                ds = json.load(ds_f)
        else:
            with open(path, 'rb') as ds_f:
                ds = pickle.load(ds_f)
        if mode == 'train':
            self.ds = ds['train'][:length]
        elif mode == 'val':
            self.ds = ds['val'][:length]
        elif mode == 'test':
            self.ds = ds['test'][:length]
        else:
            raise ValueError(f"Incorrect mode: {mode}")

        self.mode = mode
        if length is None:
            self.len = len(self.ds)
        else:
            self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        text = self.tokenizer.encode(self.tokenizer.pad_token) * self.token_size
        content = self.ds[idx]['document'] + self.tokenizer.encode(self.tokenizer.sep_token) + self.ds[idx]['summary']
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'document': text, 'sum_idx': len(self.ds[idx]['document'])}
        return sample


# class GPT21024Dataset(Dataset):
#     def __init__(self,
#                  path: str,
#                  #  mode=Literal['train', 'test', 'val'],
#                  mode: str,                     # 'train', 'test', 'val'
#                  length=None):
#         self.tokenizer = get_model_tokenizer()
#         with open(path, 'r') as ds_f:
#             ds = json.load(ds_f)
#             if mode == 'train':
#                 self.ds = ds['train'][:length]
#             elif mode == 'val':
#                 self.ds = ds['val'][:length]
#             elif mode == 'test':
#                 self.ds = ds['test'][:length]
#             else:
#                 raise ValueError(f"Incorrect mode: {mode}")
#
#         self.mode = mode
#         if length is None:
#             self.len = len(self.ds)
#         else:
#             self.len = length
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         text = self.tokenizer.encode(self.tokenizer.pad_token) * 1024
#         content = self.tokenizer.encode(self.ds[idx]['document']) + \
#                   self.tokenizer.encode(self.tokenizer.sep_token) + \
#                   self.tokenizer.encode(self.ds[idx]['summary'])
#         text[:len(content)] = content
#         # text = self.ds[idx]
#         text = torch.tensor(text)
#         sample = {'document': text, 'sum_idx': len(self.ds[idx]['document'])}
#         return sample
#
#
#




# class GPT21024Dataset(Dataset):
#
#     def __init__(self,
#                  ds_loader: DSLoader,
#                  mode=Literal['train', 'test', 'val'],
#                  length=None):
#         self.tokenizer = get_gpt2_tokenizer()
#         self.ds_loader = ds_loader
#         if mode == 'train':
#             self.y, self.X = ds_loader.y_train, ds_loader.X_train
#         elif mode == 'val':
#             self.y, self.X = ds_loader.y_val, ds_loader.X_val
#         elif mode == 'test':
#             self.y, self.X = ds_loader.y_test, ds_loader.X_test
#         else:
#             raise ValueError(f"Incorrect mode: {mode}")
#
#         # self.ds_loader.load()
#
#         # with open(ids_file,'r') as f:
#         # if mode=='train':
#         #     self.idxs = np.array(json.load(f)['train_ids'])
#         # elif mode=='valid':
#         #     self.idxs = np.array(json.load(f)['valid_ids'])
#         # elif mode=='test':
#         #     self.idxs = np.array(json.load(f)['test_ids'])
#         # self.idxs = self.idxs -min(self.idxs)
#
#         self.mode = mode
#         if length is None:
#             self.len = len(self.y)
#         else:
#             self.len = length
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         # file_name = os.path.join(self.root_dir,str(idx)+".json")
#         # file_name = os.path.join(self.root_dir, str(idx))
#         # with open(file_name, 'r') as f:
#         #     data = json.load(f)
#
#         text = self.tokenizer.encode(self.tokenizer.pad_token) * 1024
#         content =
#         content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
#         text[:len(content)] = content
#         text = torch.tensor(text)
#         sample = {'article': text, 'sum_idx': len(data['article'])}
#         return sample
