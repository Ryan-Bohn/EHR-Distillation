"""
The data module.
Define a config to load a module.
"""

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

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

from tqdm import tqdm

from glob import glob
from collections import defaultdict

import argparse

from configs import DataConfig

class DataModule:
    def __init__():
        pass