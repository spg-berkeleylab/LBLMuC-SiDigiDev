# All of the class files for separation
# Used in the training and testing separation scripts

import uproot as ur
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class InputData(Dataset):
    """
    Creates a labeled dataset from the bibFilename and noBibFilename root files specified.
    Optionally include a transform to be applied on the data.
    """
    def __init__(self, bibFilename=None, nobibFilename=None):
        if not bibFilename:
            bibFilename = "bib-features.root"
        if not nobibFilename:
            nobibFilename = "no-bib-features.root"
        bibFile = ur.open(bibFilename)
        bibTuple = bibFile["FeaturesTree"]
        nobibFile = ur.open(nobibFilename)
        nobibTuple = nobibFile["FeaturesTree"]
        
        bib_data    = InputData.createFeaturesMatrix(bibTuple)
        signal_data = InputData.createFeaturesMatrix(nobibTuple)
        self.data = torch.vstack([signal_data, bib_data])
        self.num_sig = signal_data.shape[0]
        self.num_bib = bib_data.shape[0]
        self.labels = torch.hstack([torch.ones(self.num_sig), \
                                    torch.zeros(self.num_bib)]).long()

    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        features, label = self.data[idx, :], self.labels[idx]
        return (features, label)

    @staticmethod
    def createFeaturesMatrix(ttree):
        """
        Creates an n x 5 ndarray where n is the amount of clusters in the ttree.
        The ndarray has data along its rows.
        """
        featureBranches = ['clszx', 'clszy', 'clch', 'clthe', 'clpor']
        rows = ttree.arrays(expressions=featureBranches, library='np').values()
        features_along_cols = torch.Tensor(list(rows))
        features = torch.transpose(features_along_cols, 0, 1)
        return features

class TransformableSubset:
    """
    Takes a dataset and a set of indices and creates a subset.
    Can also be transformed by a standardized transform.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def fit(self):
        """
        Create and return a StandardScaler fitting the data in this subset.
        """
        scaler = StandardScaler()
        data_tensor = torch.Tensor(len(self.indices), self.dataset[0][0].shape[-1])
        for (row, idx) in enumerate(self.indices):
            data_tensor[row] = self.dataset[idx][0]
        scaler.fit(data_tensor)
        return scaler

class StandardTransform(object):
    """
    Takes in a mean and stdev and then standardizing the data which those stats come from,
    normalizing it to have 0 mean and unit variance.

    Args:
        mean (sequence): sequence of means of each feature.
        scale (sequence): sequence of stddevs of each feature.
    """
    def __init__(self, mean, scale):
        self.mean = torch.Tensor(mean)
        self.scale = torch.Tensor(scale)

    def __call__(self, sample):
        return (sample - self.mean) / self.scale

class Net(nn.Module):
    """
    Neural network to compute classification for BIB vs signal. 
    4-layer neural network
    5 - 5 - 3 - 2
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # CrossEntropyLoss automatically applies softmax
        return x

    #backprop is automatically made by PyTorch
