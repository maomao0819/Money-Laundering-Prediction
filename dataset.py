from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
class label_Dataset(Dataset):
    def __init__(self, df):
        """ Intialize the image dataset """
        self.labels = list(df['label'])
        self.data = df.drop(columns=['label', 'alert_key']).to_numpy().astype(np.float32)
        
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.labels)

class alert_key_Dataset(Dataset):
    def __init__(self, df):
        """ Intialize the image dataset """
        self.alert_key = list(df['alert_key'])
        self.data = df.drop(columns=['label', 'alert_key']).to_numpy().astype(np.float32)
        
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        data = self.data[index]
        alert_key = self.alert_key[index]
        return data, alert_key

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.alert_key)
