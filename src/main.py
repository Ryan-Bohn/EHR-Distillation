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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MAX_SEQ_LEN = 64
EPOCHS = 100
LR = 0.001

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = os.path.join("../saved_data/", timestamp)
os.makedirs(OUT_DIR, exist_ok=True)

def parse_args():
    return None

import torch
import torch.nn as nn
import math

class TransformerEncoderForRegression(nn.Module):
    def __init__(self, num_features, max_seq_len=320, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Feature embedding layer
        self.feature_embedding = nn.Linear(num_features, embed_dim)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True # Set batch_first to True
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression Layer
        self.regressor = nn.Linear(embed_dim, 1)

        # Sinusoidal Positional Encoding
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]

        # Truncate or pad the input sequence
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
        elif seq_len < self.max_seq_len:
            pad = torch.zeros(x.size(0), self.max_seq_len - seq_len, x.size(2)).to(x.device)
            x = torch.cat([x, pad], dim=1)

        # Feature Embedding
        x = self.feature_embedding(x)

        # Add Positional Embeddings to feature embeddings
        # Adjust positional encoding to match the batch size of x
        pos_encoding = self.pe[:, :seq_len, :].expand(x.size(0), -1, -1)
        x = x + pos_encoding

        # Passing the input through the Transformer Encoder
        encoded = self.transformer_encoder(x)

        # Apply pooling over the sequence dimension to get a single vector
        pooled = encoded.mean(dim=1)

        # Regression
        output = self.regressor(pooled)
        return output



def main():
    args = parse_args()

    train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")

    # train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl") # for quicker code testing

    print(f"Datasets loaded.\nTrain set size: {len(train_set)}\nTest set size: {len(test_set)}")

    example_df = train_set[0]["feature"]
    # Convert the DataFrame to a tensor
    example_tensor = torch.tensor(example_df.values, dtype=torch.float)
    num_features = example_tensor.shape[1]

    model = TransformerEncoderForRegression(num_features=num_features, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    
    collator = Mimic3BenchmarkMultitaskDatasetLOSTaskCollator(max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collator.collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collator.collate_fn)

    # Loss Function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Lists to record losses
    train_losses = []
    eval_losses = []

    # i, (batch_features, batch_labels) = next(enumerate(train_loader))
    # print(batch_features.shape)
    # print(model(batch_features))

    # Training and Evaluation Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} training...")
        model.train()
        total_train_loss = 0
        for batch_features, batch_labels in train_loader:
            # Move tensors to the specified DEVICE
            if batch_features is None:
                continue
            batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
            
            # print(f"Batch feature shape = {batch_features.shape}")
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels.squeeze())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Evaluation
        print(f"Epoch {epoch+1} evaluating...")
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                if batch_features is None:
                    continue
                # Move tensors to the specified DEVICE
                batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)

                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_labels.squeeze())
                total_eval_loss += loss.item()

        average_eval_loss = total_eval_loss / len(test_loader)
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