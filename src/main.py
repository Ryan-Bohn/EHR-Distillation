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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MAX_SEQ_LEN = 320
EPOCHS = 100
LR = 0.001
BATCH_SIZE = 256

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = os.path.join("../saved_data/", timestamp)
os.makedirs(OUT_DIR, exist_ok=True)

def parse_args():
    return None


def main():
    args = parse_args()

    train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")

    # train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl") # for quicker code testing

    print(f"Datasets loaded.\nTrain set size: {len(train_set)}\nTest set size: {len(test_set)}")

    example_tensor = train_set[0]["feature"]
    num_features = example_tensor.shape[1]

    model = TransformerEncoderForTimeStepWiseRegression(num_features=num_features, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    
    collator = Mimic3BenchmarkMultitaskDatasetLOSTaskCollator(max_seq_len=MAX_SEQ_LEN, method="mask")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator.collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator.collate_fn)

    # Loss Function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Lists to record losses
    train_losses = []
    eval_losses = []

    # Training and Evaluation Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} training...")
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        for batch_features, batch_key_padding_masks, batch_masks, batch_labels in train_loader:
            # batch_key_padding_masks: bool tensor, true if the position is a padding
            # batch_masks: 1 only when this position is considered in computing the loss

            # Move tensors to the specified DEVICE
            batch_features, batch_key_padding_masks, batch_masks, batch_labels = batch_features.to(DEVICE), batch_key_padding_masks.to(DEVICE), batch_masks.to(DEVICE), batch_labels.to(DEVICE)
            
            # Forward pass
            outputs = model(batch_features, batch_key_padding_masks)
            masks = batch_masks == 1
            loss = criterion(outputs[masks].squeeze(), batch_labels[masks].squeeze())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_samples += torch.sum(masks).item()

        average_train_loss = total_train_loss / total_train_samples
        train_losses.append(average_train_loss)

        # Evaluation
        print(f"Epoch {epoch+1} evaluating...")
        model.eval()
        total_eval_loss = 0
        total_eval_samples = 0
        with torch.no_grad():
            for batch_features, batch_key_padding_masks, batch_masks, batch_labels in test_loader:
                # Move tensors to the specified DEVICE
                batch_features, batch_key_padding_masks, batch_masks, batch_labels = batch_features.to(DEVICE), batch_key_padding_masks.to(DEVICE), batch_masks.to(DEVICE), batch_labels.to(DEVICE)
            
                masks = batch_masks == 1
                loss = criterion(outputs[masks].squeeze(), batch_labels[masks].squeeze())
                total_eval_loss += loss.item()
                total_eval_samples += torch.sum(masks).item()

        average_eval_loss = total_eval_loss / total_eval_samples
        eval_losses.append(average_eval_loss)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {average_train_loss:.4f}, Eval Loss: {average_eval_loss:.4f}')

        # Save losses
        with open(os.path.join(OUT_DIR, 'losses.pkl'), 'wb') as f:
            pickle.dump({'train_losses': train_losses, 'eval_losses': eval_losses}, f)

        # Save the model
        model_path = os.path.join(OUT_DIR, 'transformer_encoder_regression_model.pth')
        torch.save(model.state_dict(), model_path)
    
    print(f'Training done, all data saved to "{OUT_DIR}".')
    

if __name__ == "__main__":
    main()