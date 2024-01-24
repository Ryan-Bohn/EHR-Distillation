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
from dataset import Mimic3BenchmarkMultitaskDataset

def parse_args():
    return None


def main():
    args = parse_args()

    train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")

    print(train_set[0])
    df = train_set[0]["feature"]
    # Convert the DataFrame to a tensor
    tensor = torch.tensor(df.values, dtype=torch.float)

    # Drop the first column (corresponding to 'hour_bin')
    tensor_dropped = tensor[:, 1:]
    print(tensor_dropped.shape)

if __name__ == "__main__":
    main()