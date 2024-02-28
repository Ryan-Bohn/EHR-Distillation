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
from dataclasses import dataclass, asdict
import json

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
    Mimic3BenchmarkMultitaskDatasetLOSTaskCollator, Mimic3BenchmarkMultitaskDatasetCollator,
    SyntheticMimic3BenchmarkMultitaskDataset,
    )
from model import (
    TransformerEncoderForRegression,
    TransformerEncoderForTimeStepWiseRegression, TransformerEncoderForTimeStepWiseClassification,
    TransformerEncoderPlusMimic3BenchmarkMultitaskHeads,
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# https://github.com/pytorch/pytorch/issues/117974
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = os.path.join("../saved_data/", timestamp)


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")

    # shared arguments across all subtasks
    parser.add_argument("--no_save", action="store_true", help="no checkpoint saves")
    parser.add_argument("--verbose", action="store_true", help="to verbose")

    subparsers = parser.add_subparsers(dest="exp", required=True, help='specify experiment type, in ["fit", "distill"]')
    # experiment type identifier is gonna be the 1st positional argument
    # later, access task name with args.exp
    
    parser_fit = subparsers.add_parser(name='fit', help='fit model to dataset')
    parser_distill = subparsers.add_parser(name='distill', help='distill synthetic data')
    parser_eval = subparsers.add_parser(name='eval', help='evaluate distilled data')

    # arguments for fit task
    parser_fit.add_argument('--tasks', nargs='+', required=True, help='one or more tasks in ["ihm", "los", "pheno", "decomp"]')

    # arguments for distill task
    parser_distill.add_argument('--tasks', nargs='+', required=True, help='one or more tasks in ["ihm", "los", "pheno", "decomp"]')
    parser_distill.add_argument('--method', type=str, required=True, help='distill method in ["vanilla", "gmatch"]')

    # arguments for eval task
    
    # parse the args
    args = parser.parse_args()

    if not args.no_save:
        os.makedirs(OUT_DIR, exist_ok=True)

    if args.exp == "fit": # process fit arguments

        for task in args.tasks: # check every task name is legal
            if task not in ["ihm", "los", "pheno", "decomp"]:
                raise NotImplementedError()
        args.tasks = set(args.tasks) # convert task list to set

    elif args.exp == "distill": # process distill-only arguments

        for task in args.tasks: # check every task name is legal
            if task not in ["ihm", "los", "pheno", "decomp"]:
                raise NotImplementedError()
        args.tasks = set(args.tasks) # convert task list to set

        if args.method not in ["vanilla", "gmatch"]: # make sure distill method is leagal
            raise NotImplementedError()
        
    elif args.exp == 'eval': # process syn set eval arguments
        pass
        
    else:
        raise NotImplementedError()
    
    return args


def main_legacy(): # used to fit los task in a regression manner
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
    

def fit_ihm(args):
    MAX_SEQ_LEN = 320
    EPOCHS = 100
    LR = 1e-3
    WD = 5e-4
    DROPOUT = 0.3
    BATCH_SIZE = 256
    EMBED_DIM = 32
    NUM_HEADS = 4
    NUM_LAYERS = 3

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


def compute_loss(model, data, config, param_and_buffer_dicts=None): # TODO: find a way to refactor this. It's not elegant to be on the upmost level
    """
    data: collated batch
    param_and_buffer_dicts: if not none, model's params and buffers will be replaced in forward pass
    config: for accessing weights assigned to different subtasks
    """
    if param_and_buffer_dicts is not None:
        params, buffers = param_and_buffer_dicts
    else:
        raise NotImplementedError()
    features, padding_masks = data["features"].to(DEVICE), data["padding_masks"].to(DEVICE)
    b_size = features.size(0)

    # Forward pass
    # outputs = model(features, padding_masks)
    outputs = torch.func.functional_call(
        model, (params, buffers), args=(features, padding_masks)
    )
    
    # Backward pass
    subtask_losses = []
    # compute the following subtask losses only if corresponding key is in data dict
    if "ihm_label" in data:
        ihm_pos, ihm_mask, ihm_label = data["ihm_pos"].to(DEVICE), data["ihm_mask"].to(DEVICE), data["ihm_label"].to(DEVICE)
        # note that ihm_label here is 2 dim soft label instead of class index
        # compute ihm loss
        ihm_attend = ihm_mask == 1 # valid data points that are gonno attent loss, True if data point at corresponding position is applicable to current task
        ihm_valid_outputs = outputs["ihm"][torch.arange(b_size),ihm_pos][ihm_attend] # take out the output at index ihm_pos
        ihm_valid_labels = ihm_label[ihm_attend]
        if ihm_valid_labels.size() == ihm_valid_outputs.size(): # is soft label, apply softmax before cross_entropy
            ihm_valid_labels = ihm_valid_labels.softmax(dim=-1)
        ihm_loss = F.cross_entropy(ihm_valid_outputs, ihm_valid_labels) # F.cross_entropy expect either class probabilities or single class index as target
        subtask_losses.append(ihm_loss * config.ihm_w)

    if "los_labels" in data:
        los_labels, los_masks = data["los_labels"].to(DEVICE), data["los_masks"].to(DEVICE)
        # compute los loss
        los_attend = los_masks == 1
        los_valid_outputs = outputs["los"][los_attend].squeeze(-1) # (batch_size, seq_len, 1) before squeeze
        los_valid_labels = los_labels[los_attend] # (batch_size, seq_len)
        los_loss = F.mse_loss(los_valid_outputs, los_valid_labels)
        subtask_losses.append(los_loss * config.los_w)

    if "pheno_labels" in data:
        pheno_labels = data["pheno_labels"].to(DEVICE)
        # compute pheno loss
        pheno_valid_outputs = outputs["pheno"][:, -1, :, :] # use only the whole sequence for phenotyping
        if pheno_labels.size() == pheno_valid_outputs.size(): # is soft label, apply softmax before cross_entropy
            pheno_labels = pheno_labels.softmax(dim=-1)
        pheno_loss = 0.0
        for i in range(pheno_valid_outputs.size(1)):  # iterate over the 25 classifiers
            logits = pheno_valid_outputs[:, i, :]
            labels = pheno_labels[:, i] # here shape is either (batch_size, 2) for soft labels or simply (batch_size) for indices
            loss = F.cross_entropy(logits, labels)
            pheno_loss += loss
        subtask_losses.append(pheno_loss * config.pheno_w)

    if "decomp_labels" in data:
        decomp_labels, decomp_masks = data["decomp_labels"].to(DEVICE), data["decomp_masks"].to(DEVICE)
        # compute decomp loss
        decomp_attend = decomp_masks == 1
        decomp_valid_outputs = outputs["decomp"][decomp_attend]
        decomp_valid_labels = decomp_labels[decomp_attend]
        if decomp_valid_labels.size() == decomp_valid_outputs.size(): # is soft label, apply softmax before cross_entropy
            decomp_valid_labels = decomp_valid_labels.softmax(dim=-1)
        decomp_loss = F.cross_entropy(decomp_valid_outputs, decomp_valid_labels)
        subtask_losses.append(decomp_loss * config.decomp_w)

    loss_weighted_sum = sum(subtask_losses)
    return loss_weighted_sum

@dataclass
class VanillaDistillConfig:
    n_samples: int = 1 # number of synthetic samples TODO: more
    batch_size_syn: int = 1
    batch_size_real: int = 256 # minibatch size of real datasets
    max_seq_len: int = 320
    num_heads: int = 4
    num_layers: int = 3
    embed_dim: int = 32
    n_inner_steps: int = 10 # TODO: 50
    n_epochs: int = 100
    lr_data: float = 1e-3
    wd_data: float = 1e-4
    init_lr_model: float = 1e-3
    lr_lr_model: float = 1e-3
    min_lr_model: float = 1e-5
    ihm_w: float = 1
    los_w: float = 0 # TODO
    pheno_w: float = 0
    decomp_w: float = 0


def distill(args):

    print('Starting distilling experiment...')
    print(f'Selected method: {args.method}')
    print(f'Selected subtasks: {args.tasks}')

    if args.method == "vanilla":

        config = VanillaDistillConfig()
        print(f'Configs: {json.dumps(asdict(config))}')

        # load real datasets
        train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
        test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")
        print(f"Datasets loaded. Train set size: {len(train_set)}; Test set size: {len(test_set)}")

        # use first sample in train set as example
        example_tensor = train_set[0]["feature"]
        num_features = example_tensor.shape[1]

        # create dataloaders
        collator = Mimic3BenchmarkMultitaskDatasetCollator(max_seq_len=config.max_seq_len, tasks=args.tasks) # pass in the tasks set
        train_loader = DataLoader(train_set, batch_size=config.batch_size_real, shuffle=True, collate_fn=collator.collate_fn)
        test_loader = DataLoader(test_set, batch_size=config.batch_size_real, shuffle=False, collate_fn=collator.collate_fn)

        # create synthetic dataset
        syn_set = SyntheticMimic3BenchmarkMultitaskDataset(n_samples=config.n_samples, seq_len=config.max_seq_len, n_features=num_features, tasks=args.tasks, batch_size=config.batch_size_syn)

        # get learner model
        def get_new_model():
            model = TransformerEncoderPlusMimic3BenchmarkMultitaskHeads(
                num_features=num_features,
                max_seq_len=config.max_seq_len,
                dropout=0, # disable dropout for training synthetic data
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                embed_dim=config.embed_dim,
            ).to(DEVICE)
            return model
        
        # create optimizer for synthetic data
        lr_model = torch.tensor(config.init_lr_model, dtype=torch.float, requires_grad=True)
        grouped_params = [
            {"params": syn_set.feature_list, "lr": config.lr_data, "weight_decay": config.wd_data},
            {"params": syn_set.ihm_label_list, "lr": config.lr_data, "weight_decay": config.wd_data},
            {"params": syn_set.los_time_list, "lr": config.lr_data, "weight_decay": config.wd_data},
            {"params": syn_set.pheno_labels_list, "lr": config.lr_data, "weight_decay": config.wd_data},
            {"params": syn_set.decomp_labels_list, "lr": config.lr_data, "weight_decay": config.wd_data},
            {"params": [lr_model], "lr": config.lr_lr_model}
        ]
        optimizer_data = optim.Adam(grouped_params)
        
        loss_real_e = []
        for e in range(config.n_epochs): # epoch
            print(f'---------------epoch {e+1}---------------')
            loss_real_o = []
            for o, batch_real in enumerate(train_loader): # outer loop

                model = get_new_model()
                model.train()

                # get params and buffers outside of module, so they are not constrained by pytorch
                # you can then replace them with some non-leaf tensors with complicated comp graphs
                # just use functional_call to do forward pass
                params = dict(model.named_parameters())
                buffers = dict(model.named_buffers())

                for i in range(config.n_inner_steps): # inner loop
                    # get the i-th minibatch of distilled data
                    raw_batch_syn = syn_set.get_minibatch(i)
                    batch_syn = collator.collate_fn(raw_batch_syn)
                    loss_syn = compute_loss(model, batch_syn, config, (params, buffers))
                    # update model's params manually
                    for name in params.keys():
                        param = params[name]
                        grad, = torch.autograd.grad(loss_syn, param, create_graph=True)
                        new_param = param - lr_model * grad
                        params[name] = new_param
                
                loss_real = compute_loss(model, batch_real, config, (params, buffers))
                optimizer_data.zero_grad()
                loss_real.backward()
                optimizer_data.step()
                # lr_model can't be negative
                lr_model.data.clamp_(min=config.min_lr_model)
                print(f'outer step {o+1}, avg real loss = {loss_real.item():.4f}, lr_model = {lr_model.item():.8f}')
                loss_real_o.append(loss_real.item())

            loss_real_e.append(sum(loss_real_o) / len(loss_real_o))
            print(f'! epoch {e+1} completed, avg real loss over this epoch: {loss_real_e[-1]:.4f}')
            if not args.no_save:

                # save distilled data
                current_epoch_save_path = os.path.join(OUT_DIR, f'e{e+1}_loss={loss_real_e[-1]:.4f}.pkl')
                # with open(current_epoch_save_path, 'wb') as f:
                #     pickle.dump(syn_set, f)
                # print(f"! synthetic dataset at epoch {e+1} saved as {current_epoch_save_path}")
                syn_set_state_dict = syn_set.get_state_dict()
                syn_set_state_dict["lr_model"] = lr_model.item()
                with open(current_epoch_save_path, 'wb') as f:
                    pickle.dump(syn_set_state_dict, f)
                print(f"! synthetic dataset at epoch {e+1} saved as {current_epoch_save_path}")

                # save training curves
                with open(os.path.join(OUT_DIR, 'curves.json'), 'w') as f:
                    json.dump({
                        'loss_real_e': loss_real_e,
                        }, f)
    else:
        raise NotImplementedError() # TODO



def eval(args):
    # support glob, get the lexi largest one if multiple matches
    state_dict_path = '../saved_data/20240220-235041/e42_*.pkl'
    path_match = glob.glob(state_dict_path)
    if len(path_match) < 1:
        raise Exception(f'No synthetic set state dict file found at "{state_dict_path}"')
    path_match.sort()
    true_path = path_match[-1]

    with open(true_path, 'rb') as f:
        state_dict = pickle.load(f)
    syn_set = SyntheticMimic3BenchmarkMultitaskDataset.from_state_dict(state_dict, requires_grad=False)
    lr = state_dict['lr_model']
    
    config = VanillaDistillConfig()
    model = TransformerEncoderPlusMimic3BenchmarkMultitaskHeads(
        num_features=state_dict['n_features'],
        max_seq_len=state_dict['seq_len'],
        dropout=0,
        num_heads=4,
        num_layers=3,
        embed_dim=32,
    ).to(DEVICE)

    # load real datasets
    # train_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/train/saves/*.pkl") # if passing a glob, it'll load the latest save satisfying the glob
    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/*.pkl")
    print(f"Datasets loaded. Test set size: {len(test_set)}")

    # use first sample in train set as example
    example_tensor = test_set[0]["feature"]
    num_features = example_tensor.shape[1]

    # create dataloaders
    collator = Mimic3BenchmarkMultitaskDatasetCollator(max_seq_len=config.max_seq_len, tasks={'ihm', 'los', 'pheno', 'decomp'}) # pass in the tasks set
    # train_loader = DataLoader(train_set, batch_size=config.batch_size_real, shuffle=True, collate_fn=collator.collate_fn)
    test_loader = DataLoader(test_set, batch_size=config.batch_size_real, shuffle=False, collate_fn=collator.collate_fn)

    model.train()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for i in range(config.n_inner_steps): # inner loop
        # get the i-th minibatch of distilled data
        print(i)
        raw_batch_syn = syn_set.get_minibatch(i)
        batch_syn = collator.collate_fn(raw_batch_syn)
        loss_syn = compute_loss(model, batch_syn, config, (params, buffers))
        # update model's params manually
        for name in params.keys():
            param = params[name]
            grad, = torch.autograd.grad(loss_syn, param, retain_graph=True)
            new_param = param - lr * grad
            params[name] = new_param.detach().requires_grad_(True)
    
    # Evaluation
    print(f"Evaluating trained (on synthetic data) model on real data, ihm task only")
    criterion = nn.CrossEntropyLoss()
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
            outputs = torch.func.functional_call(
                model, (params, buffers), args=(features, padding_masks)
            )['ihm']
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
    eval_auroc_score = roc_auc_score(eval_true_labels, eval_preds)

    print(f"---------Results----------")
    print(f"Eval samples: {total_eval_samples}")
    print(f"Eval loss: {average_eval_loss:.4f}")
    print(f"Eval AUROC score: {eval_auroc_score:.4f}")
    

def foo():
    # create synthetic dataset
    syn_set = SyntheticMimic3BenchmarkMultitaskDataset(n_samples=1, seq_len=320, n_features=17, tasks={"ihm", "los"}, batch_size=1)
    syn_set.save("test.pkl")
    load_syn_set = SyntheticMimic3BenchmarkMultitaskDataset.load("test.pkl")


def main():
    args = parse_args()
    if args.exp == 'fit':
        fit_ihm(args) # currently only ihm single task fitting
    elif args.exp == 'distill':
        distill(args)
    elif args.exp == 'eval':
        eval(args)


if __name__ == "__main__":
    main()