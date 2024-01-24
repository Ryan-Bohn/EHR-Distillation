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
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def collate_fn(self, batch):
        batch_features = []
        batch_labels = []
        for data_dict in batch:
            feature = data_dict["feature"]
            label = data_dict["label"]
            feature = torch.tensor(feature.values, dtype=torch.float)
            los_masks = label["los"]["masks"]
            los_labels = label["los"]["labels"]
            for t in range(len(los_masks)):
                if los_masks[t] == 1:
                    # Pad features up to the current time bin
                    padded_features = torch.nn.functional.pad(feature[:t+1], (0, 0, 0, self.max_seq_len - (t+1)))
                    batch_features.append(padded_features)
                    batch_labels.append(los_labels[t])
        # Stack all sequences and labels
        batch_features = torch.stack(batch_features)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)
        # Apply positional encoding here if not done in __getitem__
        return batch_features, batch_labels