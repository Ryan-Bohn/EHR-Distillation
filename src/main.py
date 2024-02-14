import os
import sys
import time
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.metrics import roc_auc_score
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
from dataset import (
    Mimic3BenchmarkMultitaskDataset,
    Mimic3BenchmarkMultitaskDatasetLOSTaskCollator, Mimic3BenchmarkMultitaskDatasetCollator
    )
from model import (
    TransformerEncoderForRegression,
    TransformerEncoderForTimeStepWiseRegression, TransformerEncoderForTimeStepWiseClassification
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MAX_SEQ_LEN = 320
EPOCHS = 100
LR = 1e-3
WD = 5e-4
<<<<<<< HEAD
DROPOUT = 0.3
=======
DROPOUT = 0.1
>>>>>>> 0773d5f (update doc)
BATCH_SIZE = 256
EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 3


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = os.path.join("../saved_data/", timestamp)


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    
    parser.add_argument("--no_save", action="store_true", help="no checkpoint saves")
    parser.add_argument("--exp_type", type=str, required=True, help='experiment type in ["fit", "distill"]')

    # argument group for fit exp
    fit_group = parser.add_argument_group('fit experiment arguments', 'these arguments are only available when experiment type is fit')
    fit_group.add_argument('--tasks', nargs='+', help='one or more tasks in ["ihm", "los", "pheno", "decomp"]')


    args = parser.parse_args()

    if not args.no_save:
        os.makedirs(OUT_DIR, exist_ok=True)

    if args.exp_type == "fit": # process fit-only arguments
        if not args.tasks:
            raise ValueError("select the tasks to fit a multitask model")
        for task in args.tasks: # check every task name is legal
            if task not in ["ihm", "los", "pheno", "decomp"]:
                raise NotImplementedError()
        args.tasks = set(args.tasks) # convert task list to set
    elif args.exp_type == "distill": # process distill-only arguments
        pass
    else:
        raise NotImplementedError()
    
    return args


def main():
    args = parse_args()

    if args.exp_type != "fit":
        raise NotImplementedError()
    
    # load datasets
    train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")
    print(f"Datasets loaded. Train set size: {len(train_set)}; Test set size: {len(test_set)}")

    # use first sample in train set as example
    example_tensor = train_set[0]["feature"]
    num_features = example_tensor.shape[1]

    # create dataloaders
    collator = Mimic3BenchmarkMultitaskDatasetCollator(max_seq_len=MAX_SEQ_LEN, tasks=args.tasks)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator.collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator.collate_fn)
    
    # TODO

    # loss functions
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
                
                # Forward pass
                outputs = model(batch_features, batch_key_padding_masks)
                masks = batch_masks == 1
                loss = criterion(outputs[masks].squeeze(), batch_labels[masks].squeeze())
                total_eval_loss += loss.item()
                total_eval_samples += torch.sum(masks).item()

        average_eval_loss = total_eval_loss / total_eval_samples
        eval_losses.append(average_eval_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {average_train_loss:.4f} ({total_train_samples} samples), Eval Loss: {average_eval_loss:.4f} ({total_eval_samples} samples)")

        if not args.no_save:
            # Save losses
            with open(os.path.join(OUT_DIR, 'losses.pkl'), 'wb') as f:
                pickle.dump({'train_losses': train_losses, 'eval_losses': eval_losses}, f)

            # Save the model
            model_path = os.path.join(OUT_DIR, 'transformer_encoder_regression_model.pth')
            torch.save(model.state_dict(), model_path)

    print(f'Training done, all data saved to "{OUT_DIR}".')


def main_legacy():
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
                
                # Forward pass
                outputs = model(batch_features, batch_key_padding_masks)
                masks = batch_masks == 1
                loss = criterion(outputs[masks].squeeze(), batch_labels[masks].squeeze())
                total_eval_loss += loss.item()
                total_eval_samples += torch.sum(masks).item()

        average_eval_loss = total_eval_loss / total_eval_samples
        eval_losses.append(average_eval_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {average_train_loss:.4f} ({total_train_samples} samples), Eval Loss: {average_eval_loss:.4f} ({total_eval_samples} samples)")

        if not args.no_save:
            # Save losses
            with open(os.path.join(OUT_DIR, 'losses.pkl'), 'wb') as f:
                pickle.dump({'train_losses': train_losses, 'eval_losses': eval_losses}, f)

            # Save the model
            model_path = os.path.join(OUT_DIR, 'transformer_encoder_regression_model.pth')
            torch.save(model.state_dict(), model_path)
    
    print(f'Training done, all data saved to "{OUT_DIR}".')
    

def fit_ihm():
    args = parse_args()

    if args.exp_type != "fit":
        raise NotImplementedError()

    print("Starting experiment...")
    # print hyper params
    print(f"MAX_SEQ_LEN = {MAX_SEQ_LEN}")
    print(f"EPOCHS = {EPOCHS}")
    print(f"LR = {LR}")
    print(f"WD = {WD}")
    print(f"DROPOUT = {DROPOUT}")
    print(f"BATCH_SIZE = {BATCH_SIZE}")
    print(f"EMBED_DIM = {EMBED_DIM}")
    print(f"NUM_HEADS = {NUM_HEADS}")
    print(f"NUM_LAYERS = {NUM_LAYERS}")
    
    # load datasets
    train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")
    print(f"Datasets loaded. Train set size: {len(train_set)}; Test set size: {len(test_set)}")

    # use first sample in train set as example
    example_tensor = train_set[0]["feature"]
    num_features = example_tensor.shape[1]

    # create dataloaders
    collator = Mimic3BenchmarkMultitaskDatasetCollator(max_seq_len=MAX_SEQ_LEN, tasks={"ihm"})
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator.collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator.collate_fn)
    
    # get learner model
    model = TransformerEncoderForTimeStepWiseClassification(
        num_features=num_features,
        num_classes=2,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM
        ).to(DEVICE)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Lists to record losses
    train_losses = []
    train_auroc_scores = []
    eval_losses = []
    eval_auroc_scores = []

    # Training and Evaluation Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} training...")
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        train_preds = []
        train_true_labels = []
        for data in train_loader:
            # batch_key_padding_masks: bool tensor, true if the position is a padding
            # batch_masks: 1 only when this position is considered in computing the loss

            # Move tensors to the specified DEVICE
            features, padding_masks, ihm_pos, ihm_mask, ihm_label = (
                data["features"].to(DEVICE),
                data["padding_masks"].to(DEVICE),
                data["ihm_pos"].to(DEVICE),
                data["ihm_mask"].to(DEVICE),
                data["ihm_label"].to(DEVICE),
                )
            
            b_size = features.size(0)
            # Forward pass
            outputs = model(features, padding_masks)
            valid_idx = ihm_mask == 1 # valid data point masks, True if data point at corresponding position is applicable to current task
            valid_outputs = outputs[torch.arange(b_size),ihm_pos][valid_idx]
            valid_labels = ihm_label[valid_idx]
            loss = criterion(valid_outputs, valid_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_samples += torch.sum(valid_idx).item()

            # log data for computing auroc score
            probs = torch.softmax(valid_outputs.detach(), dim=1) # Apply softmax to logits
            pos_cls_probs = probs[:, 1]
            # Store predictions and labels
            train_preds.extend(pos_cls_probs.cpu().numpy())
            train_true_labels.extend(valid_labels.cpu().numpy())

        average_train_loss = total_train_loss / total_train_samples
        train_losses.append(average_train_loss)
        train_auroc_score = roc_auc_score(train_true_labels, train_preds)
        train_auroc_scores.append(train_auroc_score)

        # Evaluation
        print(f"Epoch {epoch+1} evaluating...")
        model.eval()
        total_eval_loss = 0
        total_eval_samples = 0
        eval_preds = []
        eval_true_labels = []
        with torch.no_grad():
            for data in test_loader:
                # Move tensors to the specified DEVICE
                features, padding_masks, ihm_pos, ihm_mask, ihm_label = (
                data["features"].to(DEVICE),
                data["padding_masks"].to(DEVICE),
                data["ihm_pos"].to(DEVICE),
                data["ihm_mask"].to(DEVICE),
                data["ihm_label"].to(DEVICE),
                )

                b_size = features.size(0)
                # Forward pass
                outputs = model(features, padding_masks)
                valid_idx = ihm_mask == 1 # valid data point masks, True if data point at corresponding position is applicable to current task
                valid_outputs = outputs[torch.arange(b_size),ihm_pos][valid_idx]
                valid_labels = ihm_label[valid_idx]

                loss = criterion(valid_outputs, valid_labels)
                total_eval_loss += loss.item()
                total_eval_samples += torch.sum(valid_idx).item()

                # log data for computing auroc score
                probs = torch.softmax(valid_outputs.detach(), dim=1) # Apply softmax to logits
                pos_cls_probs = probs[:, 1]
                # Store predictions and labels
                eval_preds.extend(pos_cls_probs.cpu().numpy())
                eval_true_labels.extend(valid_labels.cpu().numpy())

        average_eval_loss = total_eval_loss / total_eval_samples
        eval_losses.append(average_eval_loss)
        eval_auroc_score = roc_auc_score(eval_true_labels, eval_preds)
        eval_auroc_scores.append(eval_auroc_score)

        print(f"---------Epoch [{epoch+1}/{EPOCHS}]----------")
        print(f"Train samples: {total_train_samples}")
        print(f"Train loss: {average_train_loss:.4f}")
        print(f"Train AUROC score: {train_auroc_score:.4f}")
        print(f"Eval samples: {total_eval_samples}")
        print(f"Eval loss: {average_eval_loss:.4f}")
        print(f"Eval AUROC score: {eval_auroc_score:.4f}")

        if not args.no_save:
            # Save losses
            with open(os.path.join(OUT_DIR, 'losses.pkl'), 'wb') as f:
                pickle.dump({
                    'train_losses': train_losses,
                    'eval_losses': eval_losses,
                    'train_auroc_scores': train_auroc_scores,
                    'eval_auroc_scores': eval_auroc_scores,
                    }, f)

            # Save the model
            model_path = os.path.join(OUT_DIR, 'transformer_encoder_regression_model.pth')
            torch.save(model.state_dict(), model_path)
    
    print(f'Training done, all data saved to "{OUT_DIR}".')
    best_train_epoch, best_train_score = max(enumerate(train_auroc_scores), key=lambda x: x[1])
    best_train_epoch += 1
    best_eval_epoch, best_eval_score = max(enumerate(eval_auroc_scores), key=lambda x: x[1])
    best_eval_epoch += 1
    print(f'Best training score: {best_train_score} (epoch {best_train_epoch})')
    print(f'Best evaluation score: {best_eval_score} (epoch {best_eval_epoch})')


def distill_multitask():
    pass # TODO

if __name__ == "__main__":
    fit_ihm()