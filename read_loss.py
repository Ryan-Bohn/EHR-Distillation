import pickle
import os

with open(os.path.join("saved_data", "20240124-133003", 'losses.pkl'), 'rb') as f:
    data_dict = pickle.load(f)

print(data_dict)