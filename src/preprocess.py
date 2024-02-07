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

import argparse

import glob
from datetime import datetime
from collections import defaultdict


def one_hot_encode(column:pd.Series, num_classes:int):
    """
    Encode a pandas series (categorical feature column with integer labels starting from 0) into one-hot representation.
    Return a dataframe.
    """
    one_hot_df = pd.DataFrame(np.zeros((len(column), num_classes), dtype=int), columns=[f'{column.name}_{i}' for i in range(num_classes)])
    for i in range(num_classes):
        one_hot_df.iloc[:, i] = (column == i).astype(int)
    return one_hot_df

class Mimic3BenchmarkDatasetPreprocessor:
    # prepare priori information about mimic3 benchmark dataset (from the original benchmark paper)
    # mimic3 benchmark dataset feature list: [(name, mean_value_from_paper, is_numeric]
    feature_list = [
        ("capillary_refill_rate", "0.0", "categorical"),
        ("diastolic_blood_pressure", 59.0, "continuous"),
        ("fraction_inspired_oxygen", 0.21, "continuous"),
        ("glascow_coma_scale_eye_opening", "4 spontaneously", "categorical"),
        ("glascow_coma_scale_motor_response", "6 obeys commands", "categorical"),
        ("glascow_coma_scale_total", "15", "categorical"),
        ("glascow_coma_scale_verbal_response", "5 oriented", "categorical"),
        ("glucose", 128.0, "continuous"),
        ("heart_rate", 86, "continuous"),
        ("height", 170.0, "continuous"),
        ("mean_blood_pressure", 77.0, "continuous"),
        ("oxygen_saturation", 98.0, "continuous"),
        ("respiratory_rate", 19, "continuous"),
        ("systolic_blood_pressure", 118.0, "continuous"),
        ("temperature", 36.6, "continuous"),
        ("weight", 81.0, "continuous"),
        ("ph", 7.4, "continuous"),
    ]
    # form it into dictionary
    feature_dict = {}
    for name, avg, ftype in feature_list:
        feature_dict[name] = {
            "avg": avg,
            "type": ftype,
            "num_cls": {
                "capillary_refill_rate": 2,
                "glascow_coma_scale_eye_opening": 4,
                "glascow_coma_scale_motor_response": 6,
                "glascow_coma_scale_total": 13,
                "glascow_coma_scale_verbal_response": 5,
            }.get(name, None),
            "bound": None,
        }
    # for some columns that may contain abnormally big /small values, add boundaries
    feature_dict["weight"]["bound"] = (0, 300)

    @staticmethod
    def map_categorical_to_numeric(name, value):
        """
        Map categorical string labels that contain certain substring to integer labels (starting from 0, np.nan for unknown)
        """

        if pd.isna(value):
            return np.nan
        substring_mapping = {
            # "capillary_refill_rate": 0.0 and 1.0
            "glascow_coma_scale_eye_opening": {
                "respon": 0,
                "pain": 1,
                "speech": 2,
                "spont": 3,
            },
            "glascow_coma_scale_motor_response": {
                "respon": 0,
                "extens": 1,
                "flex": 2,
                "withd": 3,
                "pain": 4,
                "obey": 5,
            },
            # "glascow_coma_scale_total": 3.0 to 15.0
            "glascow_coma_scale_verbal_response": {
                "respon": 0,
                "trach": 0,
                "incomp": 1,
                "inap": 2,
                "conf": 3,
                "orient": 4,
            },
        }
        if name == "capillary_refill_rate":
            return int(float(value))
        elif name == "glascow_coma_scale_total":
            return int(float(value)) - 3
        elif name in substring_mapping.keys():
            for substring, label in substring_mapping[name].items():
                if substring in value.lower():
                    return label
            return np.nan
        else: # this is not a categorical column at all
            return value
        
    def __init__(self, sr=1., tensor=True, one_hot=False, strict=False):
        self.sr = sr
        self.tensor = tensor # whether or not save processed features as torch tensor
        self.one_hot = one_hot # whether or not expand categorical feature columns (integer labels) into one-hot representations
        self.strict = strict # if set to False, samples with exception will be ignored and not saved to output; if True, exception will be raised
        

    def preprocess(self, dir, stats:dict=None):
        """
        If stats dictionary is None, compute avgs and stds (used for imputation and normalization) of the current set;
        Else, use the external stats dictionary (that probably comes from another set) to impute and normalize.
        """
        data = []
        print(f"Start preprocessing data under {dir}")
        episode_paths = glob.glob(os.path.join(dir, "*_episode*_timeseries.csv"))
        dfs = []
        keys = []
        labels = []
        # read listfile
        listfile_path = os.path.join(dir, "listfile.csv")
        listfile_dict = pd.read_csv(listfile_path, index_col=0).to_dict(orient="index")
        # read all dataframes into dfs, fill all nan cells and convert categorical to numeric
        print("Reading all time series csv files...")
        for i, path in enumerate(episode_paths):
            try:
                re_match = re.match(r"(\d+)_episode(\d+)_timeseries.csv", os.path.basename(path))
                if not re_match:
                    raise ValueError(f"Error parsing csv file: {path}")
                subject_id, episode_number = map(int, re_match.groups())
                key = f"{subject_id}_episode{episode_number}_timeseries.csv"

                # read feature csv as pandas dataframe
                df = pd.read_csv(path)

                # read all labels and masks
                if key not in listfile_dict.keys(): # check if labels are recorded in listfile
                    raise KeyError(f"Mapping key not foound: {key}")
                label_dict_raw = listfile_dict[key] # dict_keys(['length of stay', 'in-hospital mortality task (pos;mask;label)', 'length of stay task (masks;labels)', 'phenotyping task (labels)', 'decompensation task (masks;labels)'])
                los = label_dict_raw["length of stay"]
                ihm_labels_raw = label_dict_raw["in-hospital mortality task (pos;mask;label)"]
                los_labels_raw = label_dict_raw["length of stay task (masks;labels)"]
                pheno_labels_raw = label_dict_raw["phenotyping task (labels)"]
                decomp_labels_raw = label_dict_raw["decompensation task (masks;labels)"]
                ihm_pos, ihm_mask, ihm_label = [int(x) for x in ihm_labels_raw.split(';')]
                los_labels_split = los_labels_raw.split(';')
                los_masks, los_labels = [int(x) for x in los_labels_split[:len(los_labels_split)//2]], [float(x) for x in los_labels_split[len(los_labels_split)//2:]]
                pheno_labels = [int(x) for x in pheno_labels_raw.split(';')]
                decomp_labels_split = decomp_labels_raw.split(';')
                decomp_masks, decomp_labels = [int(x) for x in decomp_labels_split[:len(decomp_labels_split)//2]], [float(x) for x in decomp_labels_split[len(decomp_labels_split)//2:]]
                label_dict = {
                    "ihm": {
                        "pos": ihm_pos,
                        "mask": ihm_mask,
                        "label": ihm_label,
                    },
                    "los": {
                        "time": los,
                        "masks": los_masks,
                        "labels": los_labels,
                    },
                    "pheno": {
                        "labels": pheno_labels,
                    },
                    "decomp": {
                        "masks": decomp_masks,
                        "labels": decomp_labels,
                    },
                }
        
                # prepare features for processing
                # rename feature columns
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                # get rid of out-of-boundary numeric values by replacing with nan
                for name in Mimic3BenchmarkDatasetPreprocessor.feature_dict.keys():
                    if self.feature_dict[name]["type"] == "continuous" and self.feature_dict[name]["bound"] is not None:
                        lo, hi = self.feature_dict[name]["bound"]
                        df.loc[(df[name] < lo) | (df[name] > hi), name] = np.nan
                # map categorical features to numeric labels
                for name in self.feature_dict.keys():
                    if self.feature_dict[name]["type"] == "categorical":
                        df[name] = df[name].map(lambda x: self.map_categorical_to_numeric(name, x))
            except Exception as e:
                print(f"Something is off ({type(e)}) reading {key}, skipping...")
                if self.strict:
                    raise e
                continue
            # append all info into lists
            keys.append(key)
            dfs.append(df)
            labels.append(label_dict)
        
        # exit if no valid sample is read
        if len(keys) == 0:
            print("No valid sample processed. Exit.")
            return
        
        # compute original statistics: feature avg for imputing
        
        external_stats_flag = False
        if stats is not None:
            external_stats_flag = True
            print("Using external statistics")
        else:
            print("Computing original dataset statistics (avgs) before imputing from all valid events (lab test results)...")
        
        if not external_stats_flag:
            stats = {}
            df_in_one = pd.concat(dfs)
            df_in_one.drop(columns=["hours"], inplace=True)
            stats["orig_feature_avg"] = df_in_one.mean().to_dict()
            # if some feature avg is still nan, replace it with data reported in mimic3 benchmark paper
            for name in stats["orig_feature_avg"].keys():
                if pd.isna(stats["orig_feature_avg"][name]):
                    if self.feature_dict[name]["type"] == "continuous":
                        stats["orig_feature_avg"][name] = self.feature_dict[name]["avg"]
                    else: # categorical
                        stats["orig_feature_avg"][name] = self.map_categorical_to_numeric(name, self.feature_dict[name]["avg"])

        # resample
        print(f"Resampling to {self.sr}h bins...")
        for i, df in enumerate(dfs):
            hour_bins = np.arange(0, math.floor(labels[i]["los"]["time"]/self.sr+1)+self.sr, self.sr)
            df["hour_bin"] = pd.cut(df["hours"], bins=hour_bins, labels=False, right=False) # right is not closed
            df = df.groupby("hour_bin").last() # "hour_bin" becomes index column
            df = df.reindex(hour_bins[:-1]) # places NaN in hour bins that has no content
            df.drop(columns=["hours"], inplace=True)
            dfs[i] = df

        # get rid of nan values by 1. forward filling 2. impute with avg values
        valid_flags = [True] * len(keys)
        print("Imputing...")
        for i, df in enumerate(dfs):
            df.ffill(inplace=True) # foward fill
            for name in self.feature_dict.keys(): # replace remaining nans (at the beginning) with reported avgs in paper
                df[name].fillna(stats["orig_feature_avg"][name], inplace=True)
            if len(df) != len(labels[i]["los"]["labels"]):
                print(f'Time series "{keys[i]}" has a mismatched length with its associated los prediction task labels. Ignored.')
                print(len(df), len(labels[i]["los"]["labels"]), len(labels[i]["decomp"]["labels"]))
                valid_flags[i] = False
            # now df has to be a fully filled time series with all numeric values and no nan
            dfs[i] = df

        # before normalizing, compute statistics after imputation: feature avg & std
        if not external_stats_flag:
            print("Computing new statistics (avgs and stds) for normalizing...")
            df_in_one = pd.concat(dfs)
            stats["feature_avg"] = df_in_one.mean().to_dict()
            stats["feature_std"] = df_in_one.std().to_dict()
        
        # normalizing all columns; for categorical columns, one-hot encode them if required, or simply treat them as numeric ones and also perform normalization
        print("Z-score normalizing...")
        for i, df in enumerate(dfs):
            for name in self.feature_dict.keys():
                if self.feature_dict[name]["type"] == "continuous" or not self.one_hot:
                    df[name] = df[name] - stats["feature_avg"][name]
                    if stats["feature_std"][name] != 0:
                        df[name] = df[name] / stats["feature_std"][name]
                else: # this column is a categorical column to be one-hot encoded
                    one_hot = one_hot_encode(df[name], self.feature_dict[name]["num_cls"])
                    df.drop(name, axis=1, inplace=True)
                    df = pd.concat([df, one_hot], axis=1)
            dfs[i] = df

        # save all data into a list of dictionaries {"key": csv_file_name, "feature": tensor, "label": label_dict}
        print("Packing up data samples...")
        for i, key in enumerate(keys):
            data_dict = {
                "key": key,
                "feature": torch.tensor(dfs[i].values, dtype=torch.float) if self.tensor else dfs[i],
                "label": labels[i],
            }
            if valid_flags[i]:
                data.append(data_dict)
        # save stats and data as pickle
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(dir, "saves/")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}.pkl")
        with open(save_path, 'wb') as file:
            pickle.dump({
                "info": {
                    "sr": self.sr,
                    "one_hot": self.one_hot,
                },
                "data": data,
                "stats": stats,
            }, file)
        print(f"Finish preprocessing, pickle saved as {save_path}")
        # report some general statistics
        print(f"Processed {len(episode_paths)} csv files, {len(data)} valid time series have been loaded")
        df_lengths = np.array([len(df) for df in dfs])
        print(f'Max time series length in this dataset: {np.max(df_lengths)}')
        print(f'Average time series length in this dataset: {np.mean(df_lengths):.2f}')
        print(f'Standard deviation of time series length in this dataset: {np.std(df_lengths):.2f}')
        return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessor here.")
    
    parser.add_argument("--name", "-n", type=str, required=True, help='Dataset name. Choose from ["mimic3benchmark",]')
    parser.add_argument("--dir", "-d", type=str, required=True, help="Root directory of that chosen dataset.")
    parser.add_argument("--sr", "-s", type=float, default=1., help="Re-sampling rate in hours.")
    parser.add_argument("--one_hot", "-o", action="store_true", help="One-hot encode the catrgorical feature columns.")

    args = parser.parse_args()

    if args.name not in ["mimic3benchmark",]:
        raise NotImplementedError()
    
    if not os.path.exists(args.dir):
        raise ValueError(f"Dir {args.dir} not exists!")
    
    return args

def main():
    args = parse_args()

    if args.name == "mimic3benchmark":
        preprocessor = Mimic3BenchmarkDatasetPreprocessor(sr=args.sr, one_hot=args.one_hot)
        train_stats = preprocessor.preprocess(os.path.join(args.dir, "train/"))
        preprocessor.preprocess(os.path.join(args.dir, "test/"), stats=train_stats)

if __name__ == "__main__":
    main()