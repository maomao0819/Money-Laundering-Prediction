from torch.utils.data import Dataset

class label_Dataset(Dataset):
    def __init__(self, data, labels):
        """ Intialize the image dataset """
        self.data = data
        self.labels = labels
           
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.labels)

class alert_key_Dataset(Dataset):
    def __init__(self, data, alert_key):
        """ Intialize the image dataset """
        self.data = data
        self.alert_key = alert_key
           
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        data = self.data[index]
        alert_key = self.alert_key[index]
        return data, alert_key

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.alert_key)
