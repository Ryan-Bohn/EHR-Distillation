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