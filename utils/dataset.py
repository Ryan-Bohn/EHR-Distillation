import os
import re
import glob
import random
from collections import Counter
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np

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
    def __init__(self, dir, avg_dict, std_dict, numcls_dict, dstype="train", balance=False, mask=True, load_to_ram=True):
        """
        dir: directory of all cleaned timeseries csvs (<subject_id>_episode<#>_clean.csv) and label file (labels.csv)
        dstype: "train" or "test" 
        avg_dict and std_dict: for continous columns only
        numcls_dict: stating how many classes are there for categorical columns only
        balance: set this to be True will make every label equally distributed
        mask: set this to false will take out the mask columns
        load_to_ram: store all data in ram if set
        """
        self.dir = dir
        self.dstype = dstype
        self.avg = avg_dict
        self.std = std_dict
        self.numcls = numcls_dict
        self.mask = mask
        self.episode_paths = glob.glob(os.path.join(dir, "*_episode*_clean.csv"))
        labels_dict = pd.read_csv(os.path.join(dir, "labels.csv"), index_col=0)["y_true"].to_dict()
        # make list of labels corresponding to episode_paths
        self.labels = []
        print("Joining timeseries episodes with label...")
        for i, path in enumerate(self.episode_paths):
            re_match = re.match(r"(\d+)_episode(\d+)_clean.csv", os.path.basename(path))
            if not re_match:
                raise ValueError(f"Error parsing csv file: {path}")
            subject_id, episode_number = map(int, re_match.groups())
            key = f"{subject_id}_episode{episode_number}"
            if key not in labels_dict.keys():
                raise KeyError(f"Mapping key not foound: {key}")
            self.labels.append(labels_dict[key])
        
        self.episodes = None
        if load_to_ram:
            self.episodes = []
            print("Loading dataset to RAM...")
            unified_episodes_path = os.path.join(dir, "all_episodes.pkl")
            if os.path.exists(unified_episodes_path):
                # Loading the dictionary from the pickle file
                print(f"Found unified episodes file at {unified_episodes_path}, skipping individuals...")
                with open(unified_episodes_path, 'rb') as file:
                    episodes_dict = pickle.load(file)
                for i, path in enumerate(self.episode_paths):
                    re_match = re.match(r"(\d+)_episode(\d+)_clean.csv", os.path.basename(path))
                    if not re_match:
                        raise ValueError(f"Error parsing csv file: {path}")
                    subject_id, episode_number = map(int, re_match.groups())
                    key = f"{subject_id}_episode{episode_number}"
                    if key not in episodes_dict.keys():
                        raise KeyError(f"Mapping key: {key} not foound in unified episodes!")
                    self.episodes.append(episodes_dict[key])
            else:
                for i, path in enumerate(self.episode_paths):
                    data = pd.read_csv(path, index_col=0)
                    self.episodes.append(data)

        if balance:
            self.under_sample()

    def under_sample(self):
        # Count the number of instances in each class
        label_counts = Counter(self.labels)

        # Find the number of instances in the least common class
        min_count = label_counts[min(label_counts, key=label_counts.get)]

        # Indices to keep for each class
        indices_to_keep = {label: np.where(np.array(self.labels) == label)[0][:min_count].tolist() for label in label_counts}

        # Flatten the list of indices and sort them
        all_indices_to_keep = [idx for indices in indices_to_keep.values() for idx in indices]
        all_indices_to_keep.sort()

        # Update the episode paths and labels with the balanced dataset
        self.episode_paths = [self.episode_paths[idx] for idx in all_indices_to_keep]
        if self.episodes is not None:
            self.episodes = [self.episodes[idx] for idx in all_indices_to_keep]
        self.labels = [self.labels[idx] for idx in all_indices_to_keep]
   
    def __len__(self):
        return len(self.episode_paths)
    
    def _get_df_by_idx(self, idx):
        if self.episodes is not None:
            return self.episodes[idx]
        else:
            # get an episode file by idx
            file_path = self.episode_paths[idx]
            # read csv and return dataframe
            data = pd.read_csv(file_path, index_col=0)
            return data
    
    def __getitem__(self, idx):
        # read csv, normalize, expand categorical features to one-hot, and form a tensor sized num_features * num_time_steps
        data = self._get_df_by_idx(idx)
        processed_data = []
        for col_name, col_data in data.items():
            if "mask" in col_name: # mask column, do no processing
                if self.mask:
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
    
    def random_sample_from_class(self, n_samples, cls, no_duplicate=True, return_as_tensor=False):
        """
        Default return value: (list(samples), list(labels))
        """
        indices = [i for i, label in enumerate(self.labels) if label == cls]
        if no_duplicate:
            sampled_indices = random.sample(indices, k=n_samples)
        else:
            sampled_indices = random.choices(indices, k=n_samples)
        data_tensors = []
        label_tensors = []
        for i in sampled_indices:
            data_tensor, label_tensor = self.__getitem__(i)
            data_tensors.append(data_tensor)
            label_tensors.append(label_tensor)
        if return_as_tensor:
            return torch.stack(data_tensors, dim=0), torch.stack(label_tensors, dim=0)
        else:
            return data_tensors, label_tensors
    
    def first_n_samples_from_class(self, n_samples, cls, return_as_tensor=False):
        indices = [i for i, label in enumerate(self.labels) if label == cls]
        sampled_indices = indices[:n_samples]
        data_tensors = []
        label_tensors = []
        for i in sampled_indices:
            data_tensor, label_tensor = self.__getitem__(i)
            data_tensors.append(data_tensor)
            label_tensors.append(label_tensor)
        if return_as_tensor:
            return torch.stack(data_tensors, dim=0), torch.stack(label_tensors, dim=0)
        else:
            return data_tensors, label_tensors
        

class MultitaskPreliminaryDatasetReal(Dataset):
    # load cleaned mimic3benchmark preprocessed data as dataset
    def __init__(self, dir, avg_dict, std_dict, numcls_dict, dstype="train"):
        """
        dir: directory of all cleaned timeseries csvs (<subject_id>_episode<#>_clean.csv) and label file (labels.csv)
        dstype: "train" or "test" 
        avg_dict and std_dict: for continous columns only
        numcls_dict: stating how many classes are there for categorical columns only
        balance: set this to be True will make every label equally distributed
        mask: set this to false will take out the mask columns
        load_to_ram: store all data in ram if set
        """
        self.dir = dir
        self.dstype = dstype
        self.avg = avg_dict
        self.std = std_dict
        self.numcls = numcls_dict
        self.keys = []
        self.episodes = []
        self.ihm_labels = []
        self.los_labels = []
        self.pheno_labels = []
        print("Loading dataset to RAM...")
        pkl_path = os.path.join(dir, "all.pkl")
        if os.path.exists(pkl_path):
            # Loading the dictionary from the pickle file
            print(f"Loading from file {pkl_path}, skipping individuals...")
            with open(pkl_path, 'rb') as file:
                all_data_dict = pickle.load(file)
            for i, key in enumerate(all_data_dict.keys()):
                self.keys.append(key)
                self.episodes.append(all_data_dict[key]["ts"])
                self.ihm_labels.append(all_data_dict[key]["ihm"])
                self.los_labels.append(all_data_dict[key]["los"])
                self.pheno_labels.append(all_data_dict[key]["pheno"])
        else:
            raise NotImplementedError()

   
    def __len__(self):
        return len(self.keys)
    
    def _get_df_by_idx(self, idx):
        return self.episodes[idx]
    
    def __getitem__(self, idx):
        """
        return: ts (float tensor), ihm_label (0/1 long tensor), los_label (float tensor), pheno_label (0/1 array long tensor)
        """
        # read csv, normalize, expand categorical features to one-hot, and form a tensor sized num_features * num_time_steps
        data = self._get_df_by_idx(idx)
        processed_data = []
        for col_name, col_data in data.items():
            if "mask" in col_name: # mask column, do no processing
                pass
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
        return data_tensor, torch.tensor(self.ihm_labels[idx], dtype=torch.long), torch.tensor(self.los_labels[idx], dtype=torch.float), torch.tensor(self.pheno_labels[idx], dtype=torch.long)
    
    def _random_sample_from_class(self, n_samples, cls, no_duplicate=True, return_as_tensor=False):
        """
        Default return value: (list(samples), list(labels))
        TODO: this is not implemented
        """
        indices = [i for i, label in enumerate(self.labels) if label == cls]
        if no_duplicate:
            sampled_indices = random.sample(indices, k=n_samples)
        else:
            sampled_indices = random.choices(indices, k=n_samples)
        data_tensors = []
        label_tensors = []
        for i in sampled_indices:
            data_tensor, label_tensor = self.__getitem__(i)
            data_tensors.append(data_tensor)
            label_tensors.append(label_tensor)
        if return_as_tensor:
            return torch.stack(data_tensors, dim=0), torch.stack(label_tensors, dim=0)
        else:
            return data_tensors, label_tensors
        
    def random_sample_by_first_2_phenotype(self, n_samples, pheno0, pheno1, no_duplicate=True, return_as_tensor=False):
        """
        Random sampling from dataset with first 2 phenotypes matching given ones.
        """
        indices = [i for i, label in enumerate(self.pheno_labels) if label[0] == pheno0 and label[1] == pheno1]
        if no_duplicate and len(indices) >= n_samples:
            sampled_indices = random.sample(indices, k=n_samples)
        else:
            sampled_indices = random.choices(indices, k=n_samples)
        data_tensors = []
        ihm_label_tensors = []
        los_label_tensors = []
        pheno_label_tensors = []
        for i in sampled_indices:
            data_tensor, ihm_label_tensor, los_label_tensor, pheno_label_tensor = self.__getitem__(i)
            data_tensors.append(data_tensor)
            ihm_label_tensors.append(ihm_label_tensor)
            los_label_tensors.append(los_label_tensor)
            pheno_label_tensors.append(pheno_label_tensor)
        if return_as_tensor:
            return torch.stack(data_tensors, dim=0), torch.stack(ihm_label_tensors, dim=0), torch.stack(los_label_tensors, dim=0), torch.stack(pheno_label_tensors, dim=0)
        else:
            return data_tensors, ihm_label_tensors, los_label_tensors, pheno_label_tensors
        

