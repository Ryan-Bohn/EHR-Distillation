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
    # # Path to the pickle file
    # filepath = "../saved_data/20240130-061945/losses.pkl"

    # # Load the data
    # with open(filepath, 'rb') as f:
    #     data = pickle.load(f)


    # # Extract the training and evaluation losses
    # train_losses = data['train_losses']
    # eval_losses = data['eval_losses']

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(eval_losses, label='Evaluation Loss')
    # plt.title('Training and Evaluation Losses')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Data from the provided JSON content
    # loss_real_e = [21.7503724818607, 20.198674236270165, 18.581420171175072, 17.132102739896705, 15.847707563166995, 
    #             14.70021234141837, 13.849141381627364, 13.145135948126265, 12.61773599995126, 12.224841961757742, 
    #             11.921610996877547, 11.667429306524262, 11.492391147201868, 11.350556126601404, 11.245595088107981]

    # # Creating the plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(loss_real_e, marker='o', color='b')
    # plt.title("Training Loss Curve")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.grid(True)
    # plt.show()

    # plot synthetic dataset
    fp_glob = "../saved_data/20240228-055445/e65*.pkl"
    fp_list = glob.glob(fp_glob)
    fp = fp_list[0]
    print(fp)
    with open(fp, 'rb') as f:
        data_dict = pickle.load(f)
    print(data_dict.keys())
    print(data_dict["n_samples"])
    print(data_dict["feature_list"][0].shape)
    data = data_dict["feature_list"][0]

    test_set = Mimic3BenchmarkMultitaskDataset("../data/mimic3/benchmark/multitask/test/saves/20240214-*.pkl")

    real_data = test_set[0]["feature"]
    print(real_data.shape)

    # Plotting the real timeseries
    plt.figure(figsize=(12, 6))
    for i in range(real_data.shape[1]):
        plt.plot(real_data[:48, i], label=f'Feature {i+1}')

    plt.xlabel('Time (0~47h)')
    plt.ylabel('Values')
    plt.title('Real sample illustration')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    plt.tight_layout()
    plt.show()

    # Plotting the 17-variate timeseries
    plt.figure(figsize=(12, 6))
    for i in range(data.shape[1]):
        plt.plot(data[:48, i], label=f'Feature {i+1}')

    plt.xlabel('Time (0~47h)')
    plt.ylabel('Values')
    plt.title('Distilled sample illustration')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()