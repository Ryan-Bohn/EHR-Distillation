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
from dataset import Mimic3BenchmarkMultitaskDataset, Mimic3BenchmarkMultitaskDatasetLOSTaskCollator
from model import TransformerEncoderForRegression, TransformerEncoderForTimeStepWiseRegression

def parse_args():
    return None


def main():
    # Path to the pickle file
    filepath = "../saved_data/20240130-061945/losses.pkl"

    # Load the data
    with open(filepath, 'rb') as f:
        data = pickle.load(f)


    # Extract the training and evaluation losses
    train_losses = data['train_losses']
    eval_losses = data['eval_losses']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.title('Training and Evaluation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()