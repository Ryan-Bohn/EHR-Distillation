import os
import re
import glob
import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.detach().float()
        self.y = y.detach().long()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]
    

def tensor_to_preliminary_timeseries(x: torch.Tensor):
    pass


class IHMPreliminaryDatasetReal(Dataset):
    # load cleaned mimic3benchmark preprocessed data as dataset
    def __init__(self, dir, avg_dict, std_dict, numcls_dict, dstype="train"):
        """
        dir: directory of all cleaned timeseries csvs (<subject_id>_episode<#>_clean.csv) and label file (labels.csv)
        dstype: "train" or "test" 
        avg_dict and std_dict: for continous columns only
        numcls_dict: stating how many classes are there for categorical columns only
        """
        self.dir = dir
        self.dstype = dstype
        self.avg = avg_dict
        self.std = std_dict
        self.numcls = numcls_dict
        self.episode_paths = glob.glob(os.path.join(dir, "*_episode*_clean.csv"))
        labels_dict = pd.read_csv(os.path.join(dir, "labels.csv"), index_col=0)["y_true"].to_dict()
        # make list of labels corresponding to episode_paths
        self.labels = []
        for i, path in enumerate(self.episode_paths):
            re_match = re.match(r"(\d+)_episode(\d+)_clean.csv", os.path.basename(path))
            if not re_match:
                raise ValueError(f"Error parsing csv file: {path}")
            subject_id, episode_number = map(int, re_match.groups())
            key = f"{subject_id}_episode{episode_number}"
            if key not in labels_dict.keys():
                raise KeyError(f"Mapping key not foound: {key}")
            self.labels.append(labels_dict[key])
   
    def __len__(self):
        return len(self.episode_paths)
    
    def __getitem__(self, idx):
        # get an episode file by idx
        file_path = self.episode_paths[idx]

        # read csv, normalize, expand categorical features to one-hot, and form a tensor sized num_features * num_time_steps
        data = pd.read_csv(file_path, index_col=0)
        processed_data = []
        for col_name, col_data in data.items():
            if "mask" in col_name: # mask column, do no processing
                processed_data.append(torch.tensor(col_data.values, dtype=torch.float).unsqueeze(1)) # sized (seqlen*1)
            elif col_name in self.avg.keys(): # column is a continuous feature, normalize it
                normalized_col_data = (col_data - self.avg[col_name]) / self.std[col_name]
                processed_data.append(torch.tensor(normalized_col_data.values, dtype=torch.float).unsqueeze(1)) # sized (seqlen*1)
            elif col_name in self.numcls.keys() : # column is categorical, one-hot it
                one_hot_col_data = F.one_hot(torch.tensor(col_data.astype(int).values)-1, num_classes=self.numcls[col_name]) # -1 for one_hot() expect input starting from 0
                processed_data.append(one_hot_col_data) # sized (seqlen*numcls)
            else:
                raise ValueError("Can't identify column: {col_name}")
        data_tensor = torch.cat(processed_data, dim=1) # sized (seqlen * num_features) where seqlen is 48/sample_rate_in_hour

        # load labels
        return data_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
    
    def random_sample_from_class(self, n_samples, cls):
        indices = [i for i, label in enumerate(self.labels) if label == cls]
        sampled_indices = random.sample(indices, n_samples)
        data_tensors = []
        label_tensors = []
        for i in sampled_indices:
            data_tensor, label_tensor = self.__getitem__(i)
            data_tensors.append(data_tensor)
            label_tensors.append(label_tensor)
        return torch.stack(data_tensors, dim=0), torch.stack(label_tensors, dim=0)

        

