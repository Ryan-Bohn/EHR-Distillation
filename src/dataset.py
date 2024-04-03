import os
import sys
import time
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
import copy
import pickle
import itertools
from datetime import datetime
import pandas as pd
import heapq
import re
import math

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from tqdm import tqdm

import glob
from datetime import datetime
from collections import defaultdict

import argparse

from preprocess import Mimic3BenchmarkDatasetPreprocessor


class Mimic3BenchmarkMultitaskDataset(Dataset):
    
    def __init__(self, path):
        self.stats = None
        self.data = None
        self.info = None
        save_paths = glob.glob(path)
        if len(save_paths) < 1:
            raise Exception(f'No dataset save file (.pkl) found satisfying "{path}"!')
        save_paths.sort()
        latest_save = save_paths[-1]
        self.path = latest_save
        print(f'Loading dataset from "{self.path}"...')
        with open(latest_save, 'rb') as f:
            save_dict = pickle.load(f)
            self.data = save_dict["data"]
            self.stats = save_dict["stats"]
            self.info = save_dict["info"]
        print(f"Dataset loaded. Info: {self.info}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class Mimic3BenchmarkMultitaskDatasetLOSTaskCollator:
    def __init__(self, max_seq_len, method: str):
        
        self.max_seq_len = max_seq_len

        if method == "subseq":
            self.collate_fn = self.collate_fn_subseq
        elif method == "mask":
            self.collate_fn = self.collate_fn_mask
        else:
            raise NotImplementedError()

    def collate_fn_subseq(self, batch):
        batch_features = []
        batch_labels = []
        for data_dict in batch:
            feature = data_dict["feature"]
            label = data_dict["label"]
            los_masks = label["los"]["masks"]
            los_labels = label["los"]["labels"]
            for t in range(len(los_masks)):
                if los_masks[t] == 1:
                    # Pad features up to the current time bin
                    padded_features = torch.nn.functional.pad(feature[:t+1], (0, 0, 0, self.max_seq_len - (t+1)))
                    batch_features.append(padded_features)
                    batch_labels.append(los_labels[t])
        # Stack all sequences and labels
        if len(batch_features) == 0:
            return None, None
        batch_features = torch.stack(batch_features)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)

        return batch_features, batch_labels
    
    def collate_fn_mask(self, batch):
        batch_features = []
        batch_labels = []
        batch_masks = []
        batch_key_padding_masks = []

        for data_dict in batch:
            feature = data_dict["feature"]
            label = data_dict["label"]
            los_masks = torch.tensor(label["los"]["masks"], dtype=torch.long)
            los_labels = torch.tensor(label["los"]["labels"], dtype=torch.float)

            # Pad features and labels to max_length
            
            padded_features = torch.nn.functional.pad(feature, (0, 0, 0, self.max_seq_len - feature.size(0)))
            padded_labels = torch.nn.functional.pad(los_labels, (0, self.max_seq_len - los_labels.size(0)), value=-1)  # Use a dummy value for padding
            padded_masks = torch.nn.functional.pad(los_masks, (0, self.max_seq_len - los_masks.size(0)), value=0)

            # Create a mask for valid data points
            # mask = (padded_masks == 1) & (torch.arange(self.max_seq_len) < feature.size(0))
            # mask = ~mask
            key_padding_masks = ~(torch.arange(self.max_seq_len) < feature.size(0))

            batch_features.append(padded_features)
            batch_labels.append(padded_labels)
            batch_key_padding_masks.append(key_padding_masks)
            batch_masks.append(padded_masks)

        # Stack all sequences, labels, and masks
        if len(batch_features) == 0:
            return None, None, None

        batch_features = torch.stack(batch_features)
        batch_labels = torch.stack(batch_labels)
        batch_masks = torch.stack(batch_masks)
        batch_key_padding_masks = torch.stack(batch_key_padding_masks)

        return batch_features, batch_key_padding_masks, batch_masks, batch_labels
    
class Mimic3BenchmarkMultitaskDatasetCollatorLegacy: # this collator assume raw batch has not everything in tensor, but only feature
    def __init__(self, max_seq_len, tasks: set):
        
        self.max_seq_len = max_seq_len
        self.tasks = tasks

    def collate_fn(self, raw_batch):
        """
        Returns a dictionary.
        """

        batch = {}
        
        # pack features and padding masks
        batch_features = []
        batch_padding_masks = []
        for data_dict in raw_batch:
            feature = data_dict["feature"]
            # pad feature to max_length
            padded_features = torch.nn.functional.pad(feature, (0, 0, 0, self.max_seq_len - feature.size(0)))
            batch_features.append(padded_features)
            # create a mask for valid (non-padding) data points
            padding_masks = ~(torch.arange(self.max_seq_len) < feature.size(0)) # True means this data point is a padding
            batch_padding_masks.append(padding_masks)
        if len(batch_features) == 0:
            return None
        batch_features = torch.stack(batch_features)
        batch_padding_masks = torch.stack(batch_padding_masks)
        batch["features"] = batch_features
        batch["padding_masks"] = batch_padding_masks
        
        # pack ihm related labels
        if "ihm" in self.tasks:
            batch_ihm_pos = []
            batch_ihm_mask = []
            batch_ihm_label = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                ihm_pos = torch.tensor(label["ihm"]["pos"], dtype=torch.long)
                ihm_mask = torch.tensor(label["ihm"]["mask"], dtype=torch.long)
                ihm_label = torch.tensor(label["ihm"]["label"], dtype=torch.long)
                batch_ihm_pos.append(ihm_pos)
                batch_ihm_mask.append(ihm_mask)
                batch_ihm_label.append(ihm_label)
            batch_ihm_pos = torch.stack(batch_ihm_pos)
            batch_ihm_mask = torch.stack(batch_ihm_mask)
            batch_ihm_label = torch.stack(batch_ihm_label)
            batch["ihm_pos"] = batch_ihm_pos
            batch["ihm_mask"] = batch_ihm_mask
            batch["ihm_label"] = batch_ihm_label
        
        # pack los related labels:
        if "los" in self.tasks:
            batch_los_labels = []
            batch_los_masks = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                los_masks = torch.tensor(label["los"]["masks"], dtype=torch.long)
                los_labels = torch.tensor(label["los"]["labels"], dtype=torch.float)
                # los labels and masks are of same length as feature seq, also pad them
                padded_los_labels = torch.nn.functional.pad(los_labels, (0, self.max_seq_len - los_labels.size(0)), value=-1)  # use a dummy value for padding
                padded_los_masks = torch.nn.functional.pad(los_masks, (0, self.max_seq_len - los_masks.size(0)), value=0)
                batch_los_labels.append(padded_los_labels)
                batch_los_masks.append(padded_los_masks)
            batch_los_labels = torch.stack(batch_los_labels)
            batch_los_masks = torch.stack(batch_los_masks)
            batch["los_labels"] = batch_los_labels
            batch["los_masks"] = batch_los_masks
            
        
        # pack pheno related labels:
        if "pheno" in self.tasks:
            batch_pheno_labels = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                pheno_labels = torch.tensor(label["pheno"]["labels"], dtype=torch.long)
                batch_pheno_labels.append(pheno_labels)
            batch_pheno_labels = torch.stack(batch_pheno_labels)
            batch["pheno_labels"] = batch_pheno_labels
        
        # pack decomp related labels:
        if "decomp" in self.tasks:
            batch_decomp_labels = []
            batch_decomp_masks = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                decomp_masks = torch.tensor(label["decomp"]["masks"], dtype=torch.long)
                decomp_labels = torch.tensor(label["decomp"]["labels"], dtype=torch.long)
                # decomp labels and masks are of same length as feature seq, also pad them
                padded_decomp_labels = torch.nn.functional.pad(decomp_labels, (0, self.max_seq_len - decomp_labels.size(0)), value=-1)  # use a dummy value for padding
                padded_decomp_masks = torch.nn.functional.pad(decomp_masks, (0, self.max_seq_len - decomp_masks.size(0)), value=0)
                batch_decomp_labels.append(padded_decomp_labels)
                batch_decomp_masks.append(padded_decomp_masks)
            batch_decomp_labels = torch.stack(batch_decomp_labels)
            batch_decomp_masks = torch.stack(batch_decomp_masks)
            batch["decomp_labels"] = batch_decomp_labels
            batch["decomp_masks"] = batch_decomp_masks

        return batch
    
class Mimic3BenchmarkMultitaskDatasetCollator:

    def __init__(self, max_seq_len, tasks: set):
        
        self.max_seq_len = max_seq_len
        self.tasks = tasks

    def collate_fn(self, raw_batch):
        """
        Returns a dictionary.
        """

        batch = {}
        
        # pack features and padding masks
        batch_features = []
        batch_padding_masks = []
        for data_dict in raw_batch:
            feature = data_dict["feature"]
            # truncate feature to max_seq_len if longer
            if feature.size(0) > self.max_seq_len:
                feature = feature[:self.max_seq_len]
            # pad feature to max_length
            padded_features = torch.nn.functional.pad(feature, (0, 0, 0, self.max_seq_len - feature.size(0)))
            batch_features.append(padded_features)
            # create a mask for valid (non-padding) data points
            padding_masks = ~(torch.arange(self.max_seq_len) < feature.size(0)) # True means this data point is a padding
            batch_padding_masks.append(padding_masks)
        if len(batch_features) == 0:
            return None
        batch_features = torch.stack(batch_features)
        batch_padding_masks = torch.stack(batch_padding_masks)
        batch["features"] = batch_features
        batch["padding_masks"] = batch_padding_masks
        
        # pack ihm related labels
        if "ihm" in self.tasks:
            batch_ihm_pos = []
            batch_ihm_mask = []
            batch_ihm_label = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                ihm_pos = label["ihm"]["pos"]
                ihm_mask = label["ihm"]["mask"]
                ihm_label = label["ihm"]["label"]
                batch_ihm_pos.append(ihm_pos)
                batch_ihm_mask.append(ihm_mask)
                batch_ihm_label.append(ihm_label)
            batch_ihm_pos = torch.stack(batch_ihm_pos)
            batch_ihm_mask = torch.stack(batch_ihm_mask)
            batch_ihm_label = torch.stack(batch_ihm_label)
            batch["ihm_pos"] = batch_ihm_pos
            batch["ihm_mask"] = batch_ihm_mask
            batch["ihm_label"] = batch_ihm_label
        
        # pack los related labels:
        if "los" in self.tasks:
            batch_los_labels = []
            batch_los_masks = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                los_masks = label["los"]["masks"]
                los_labels = label["los"]["labels"]
                # los labels and masks are of same length as feature seq, also pad them
                padded_los_labels = torch.nn.functional.pad(los_labels, (0, self.max_seq_len - los_labels.size(0)), value=-1)  # use a dummy value for padding
                padded_los_masks = torch.nn.functional.pad(los_masks, (0, self.max_seq_len - los_masks.size(0)), value=0)
                batch_los_labels.append(padded_los_labels)
                batch_los_masks.append(padded_los_masks)
            batch_los_labels = torch.stack(batch_los_labels)
            batch_los_masks = torch.stack(batch_los_masks)
            batch["los_labels"] = batch_los_labels
            batch["los_masks"] = batch_los_masks
            
        
        # pack pheno related labels:
        if "pheno" in self.tasks:
            batch_pheno_labels = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                pheno_labels = label["pheno"]["labels"]
                batch_pheno_labels.append(pheno_labels)
            batch_pheno_labels = torch.stack(batch_pheno_labels)
            batch["pheno_labels"] = batch_pheno_labels
        
        # pack decomp related labels:
        if "decomp" in self.tasks:
            batch_decomp_labels = []
            batch_decomp_masks = []
            for data_dict in raw_batch:
                label = data_dict["label"]
                decomp_masks = label["decomp"]["masks"]
                decomp_labels = label["decomp"]["labels"]
                # decomp labels and masks are of same length as feature seq, also pad them
                padded_decomp_labels = torch.nn.functional.pad(decomp_labels, (0, self.max_seq_len - decomp_labels.size(0)), value=-1)  # use a dummy value for padding
                padded_decomp_masks = torch.nn.functional.pad(decomp_masks, (0, self.max_seq_len - decomp_masks.size(0)), value=0)
                batch_decomp_labels.append(padded_decomp_labels)
                batch_decomp_masks.append(padded_decomp_masks)
            batch_decomp_labels = torch.stack(batch_decomp_labels)
            batch_decomp_masks = torch.stack(batch_decomp_masks)
            batch["decomp_labels"] = batch_decomp_labels
            batch["decomp_masks"] = batch_decomp_masks

        return batch
    

class SyntheticMimic3BenchmarkMultitaskDataset(Dataset):
    """
    Synthetic dataset for multitask mimic3 benchmark.
    """
    version = "1.0"

    def __init__(self, n_samples: int, seq_len: int, n_features: int, tasks: set, batch_size: int):
        
        self.n_samples = n_samples
        self.tasks = tasks
        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size

        # all learnable things as separate lists
        self.feature_list = [torch.randn((seq_len, n_features), dtype=torch.float, requires_grad=True) for _ in range(n_samples)]
        self.ihm_label_list = [torch.randn(2, dtype=torch.float, requires_grad=True) for _ in range(n_samples)] # soft labels' logits, return to learner after softmax
        self.los_time_list = [torch.randn(1, dtype=torch.float, requires_grad=True) for _ in range(n_samples)]
        self.pheno_labels_list = [torch.randn((25, 2), dtype=torch.float, requires_grad=True) for _ in range(n_samples)] # soft labels' logits, return to learner after softmax
        self.decomp_labels_list = [torch.randn((seq_len, 2), dtype=torch.float, requires_grad=True) for _ in range(n_samples)] # soft labels' logits, return to learner after softmax

        # some tensors that are not learnable, thus can be shared
        self.ihm_pos = torch.tensor(47, dtype=torch.long)
        self.ihm_mask = torch.tensor(1 if self.seq_len > 47 else 0, dtype=torch.long)
        self.first_5_masks = torch.ones(self.seq_len)
        self.first_5_masks[:5] = 0

    
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        time_past = torch.arange(self.seq_len, dtype=torch.float) # 0 to seq_len-1 hours in a tensor

        return {
            "feature": self.feature_list[idx],
            "label": {
                "ihm": {
                    "pos": self.ihm_pos,
                    "mask": self.ihm_mask,
                    "label": self.ihm_label_list[idx],
                },
                "los": {
                    "masks": self.first_5_masks,
                    "labels": self.los_time_list[idx] - time_past, # all positions are related to the one and only los time
                },
                "pheno": {
                    "labels": self.pheno_labels_list[idx],
                },
                "decomp": {
                    "masks": self.first_5_masks,
                    "labels": self.decomp_labels_list[idx],
                },
            },
        }
    
    def get_minibatch(self, batch_id):
        batch = []
        i_start = batch_id * self.batch_size % self.n_samples
        for i in range(i_start, i_start+self.batch_size):
            batch.append(self.__getitem__(i%self.n_samples))
        return batch
    
    def get_state_dict(self):
        state_dict = {
            'version': self.version,
            'n_samples': self.n_samples,
            'tasks': self.tasks,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'batch_size': self.batch_size,
            'feature_list': [t.detach().cpu().numpy() for t in self.feature_list],
            'ihm_label_list': [t.detach().cpu().numpy() for t in self.ihm_label_list],
            'los_time_list': [t.detach().cpu().numpy() for t in self.los_time_list],
            'pheno_labels_list': [t.detach().cpu().numpy() for t in self.pheno_labels_list],
            'decomp_labels_list': [t.detach().cpu().numpy() for t in self.decomp_labels_list],
            'ihm_pos': self.ihm_pos.detach().cpu().numpy(),
            'ihm_mask': self.ihm_mask.detach().cpu().numpy(),
            'first_5_masks': self.first_5_masks.detach().cpu().numpy(),
        }
        return state_dict
    
    def save(self, save_path):
        """
        Save the dataset instance to a file.
        """
        state_dict = self.get_state_dict()
        with open(save_path, 'wb') as f:
            pickle.dump(state_dict, f)

    @classmethod
    def from_state_dict(cls, state_dict, requires_grad=True):
        save_version = state_dict['version'] # use this for compatibility

        # Initialize a new instance
        new_instance = cls(
            n_samples=state_dict['n_samples'],
            tasks=state_dict['tasks'],
            seq_len=state_dict['seq_len'],
            n_features=state_dict['n_features'],
            batch_size=state_dict['batch_size']
        )

        # Restore tensor attributes
        new_instance.feature_list = [torch.tensor(t, requires_grad=requires_grad) for t in state_dict['feature_list']]
        new_instance.ihm_label_list = [torch.tensor(t, requires_grad=requires_grad) for t in state_dict['ihm_label_list']]
        new_instance.los_time_list = [torch.tensor(t, requires_grad=requires_grad) for t in state_dict['los_time_list']]
        new_instance.pheno_labels_list = [torch.tensor(t, requires_grad=requires_grad) for t in state_dict['pheno_labels_list']]
        new_instance.decomp_labels_list = [torch.tensor(t, requires_grad=requires_grad) for t in state_dict['decomp_labels_list']]
        new_instance.ihm_pos = torch.tensor(state_dict['ihm_pos'])
        new_instance.ihm_mask = torch.tensor(state_dict['ihm_mask'])
        new_instance.first_5_masks = torch.tensor(state_dict['first_5_masks'])

        return new_instance

    @classmethod
    def load(cls, save_path):
        """
        Load the dataset from a file.
        """
        with open(save_path, 'rb') as f:
            state_dict = pickle.load(f)

        return cls.from_state_dict(state_dict)
        