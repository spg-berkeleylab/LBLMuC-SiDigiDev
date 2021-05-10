## CLI Script for NN separation of BIB
## By Rohit Agarwal
#-------------------------------------
## usage: py training-separation.py <nobibfile> <bibfile> <modelfile> <paramfile>
## nobibfile - a ROOT file containing the features of a no-bib dataset
## bibfile - a ROOT file containing the features of a bib dataset
## modelfile - path to save the model. Should be a .pt file.
## paramfile - path to save the parameters. Should be a .json file.

import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import uproot as ur
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
#%matplotlib inline

LAYER = None

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

# %%
def split_data(data, training_ratio=0.5, val_ratio=0.25):
    """
    Splits the data into training, test, and validation.
    Args:
        data: An InputData object.
        training_ratio: ratio of training to total.
        val_ratio: ratio of validation to total.
    Returns:
        train_load: A DataLoader for training.
        val_load: A DataLoader for validation.
        test_load: A DataLoader for testing.
        scaler: The scaler that was used on the data
    """
    len_train, len_val = round(training_ratio * len(data)), round(val_ratio * len(data))
    len_test = len(data) - len_train - len_val
    # indices inhabited by each set
    train_ind, val_ind, test_ind = random_split(np.arange(len(data)), [len_train, len_val, len_test])

    train_ind, val_ind, test_ind = list(train_ind), list(val_ind), list(test_ind)
    train_set = TransformableSubset(data, train_ind)
    print('train_data',data.data[train_ind])
    # fit the transform to the train_set, but transform all the data
    scaler = train_set.fit()
    print('mean',scaler.mean_)
    print('scale',scaler.scale_)
    transform = StandardTransform(scaler.mean_, scaler.scale_)
    print(f"Data before normalization: {data.data}")
    data.data = transform(data.data)
    print(f"Data after normalization: {data.data}")
    #Create samplers to randomly sample the data
    train_sampler = SubsetRandomSampler(train_ind)
    val_sampler   = SubsetRandomSampler(val_ind)
    test_sampler  = SubsetRandomSampler(test_ind)
    #Use data-loaders to create iterables
    train_load = DataLoader(data, batch_size=512, sampler=train_sampler)
    val_load   = DataLoader(data, batch_size=256, sampler=val_sampler  )
    test_load  = DataLoader(data, batch_size=256, sampler=test_sampler )
    return train_load, val_load, test_load, scaler

def plot_all_features(data):
    # Plot all features
    for idx in range(data.data.shape[1]):
        plt.hist(data.data[data.labels==0,idx].numpy(),bins=100,label='1',density=True,histtype='step')
        plt.hist(data.data[data.labels==1,idx].numpy(),bins=100,label='0',density=True,histtype='step')
        plt.xlabel(f'Normalized Feature {idx}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'feature_{idx}.png')
        plt.clf()

# %%
def train_nn(classifier, optimizer, criterion, train_load, val_load, MAX_EPOCHS=50):
    """
    Trains an NN for a given amount of epochs on some data.
    Args:
        classifier: Net model.
        optimizer: Optimizer that computes gradients.
        criterion: Loss criterion.
        train_load: DataLoader for the training set.
        val_load: DataLoader for the validation set.
        MAX_EPOCHS: The amount of epochs to train for.
    Returns:
        training_losses: A list of losses throughout the epochs for training.
        validation_losses: Same as above but for validation.
    """
    # Train the NN
    training_losses   = []
    validation_losses = []
    # training loop

    train_iter = iter(train_load)
    val_iter = iter(val_load)

    for epoch in range(MAX_EPOCHS):
        # Train epoch
        running_tloss = 0
        n = 0    
        for batch, (samples_batched, labels_batched) in enumerate(train_load):
            optimizer.zero_grad()   # zero the gradient buffers
            output = classifier(samples_batched)
            loss = criterion(output, labels_batched)
            loss.backward()
            optimizer.step()
            running_tloss+=loss.item()
            n += 1
            if batch % 100 == 99:
                print('[{epoch}, {batch}] {loss}'.format(epoch=epoch,batch=batch,loss=running_tloss/n))
        running_tloss /= n
        training_losses.append(running_tloss)
                
        # Compute validation loss
        running_vloss = 0
        n = 0        
        with torch.set_grad_enabled(False):
            for batch, (samples_batched, labels_batched) in enumerate(val_load):
                output = classifier(samples_batched)
                loss = criterion(output, labels_batched)
                running_vloss += loss.item()
                n += 1
            running_vloss /= n
            validation_losses.append(running_vloss)

        print('Epoch {}: {} {}'.format(epoch, running_tloss, running_vloss))

    return training_losses, validation_losses

def plot_loss(training_losses, validation_losses):
    plt.plot(training_losses  ,label='Training')
    plt.plot(validation_losses,label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if LAYER != -1 and LAYER != None:
        plt.savefig(f'loss-layer_{LAYER}.png')
    else:
        plt.savefig('loss.png')
    plt.clf()

# %%
def check_test(classifier, test_load):
    """
    Checks a classifier on some testing data.
    Returns:
        scor0: A numpy Array of the probabilities that each hit is BIB (0).
        scor1: A numpy Array of the probability that each hit is signal (1).
        label: A numpy Array of the actual status of the hit (BIB or signal).
    """
    # Evaluate the NN on the test batch
    scor0 = []
    scor1 = []
    label = []
    torch.set_grad_enabled(False) # testing now
    for samples_batched, labels_batched in test_load:
        output = F.softmax(classifier(samples_batched), dim=-1).numpy()

        scor0.extend(output[:,0])
        scor1.extend(output[:,1])
        label.extend(labels_batched)

    scor0 = np.array(scor0)
    scor1 = np.array(scor1)
    label = np.array(label)
    return scor0, scor1, label

# %%
def plot_test_outputs(scor0, scor1, label):
    # Plot the NN classification outputs
    plt.hist(scor0[label == 0],label='BIB'   ,bins=20,range=(0,1),histtype='step',density=True)
    plt.hist(scor0[label == 1],label='Signal',bins=20,range=(0,1),histtype='step',density=True)
    plt.xlabel("Probability of BIB")
    plt.legend()
    plt.tight_layout()
    plt.savefig('probBIB.png')
    plt.clf()

    plt.hist(scor1[label == 0],label='BIB'   ,bins=20,range=(0,1),histtype='step',density=True)
    plt.hist(scor1[label == 1],label='Signal',bins=20,range=(0,1),histtype='step',density=True)
    plt.xlabel("Probability of Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig('probSignal.png')
    plt.clf()

def plot_best_cut(scor0, scor1, label, sig_eff_min=0.975):
    """
    Args:
        scor0, scor1, label: See check_test.
        sig_eff_min: The minimum signal efficiency we'd allow.
    Returns:
        prob_cutoff: The best cutoff probability above which we consider a hit as signal.
        sig_eff_cut: The efficiency of signal cutting with this cut.
        bib_eff_cut: The efficiency of bib cutting with this cut.
    """
    #Make ROC curve
    total = np.bincount(label)
    bib_eff = []
    sig_eff = []
    for cut in np.arange(0, 1, 0.01):
        results=np.bincount(label[cut<scor1], minlength=2)
        bib_eff.append(results[0] / total[0])
        sig_eff.append(results[1] / total[1])

    print(bib_eff)
    print(sig_eff)

    for i, eff in enumerate(sig_eff):
        if eff < sig_eff_min:
            best_cut = i - 1
            break
    sig_eff_cut = sig_eff[best_cut]
    bib_eff_cut = bib_eff[best_cut]
    prob_cutoff = best_cut * 0.01

    #Plot ROC curve
    plt.plot(sig_eff, bib_eff)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('BIB Efficiency')
    plt.axvline(x=sig_eff_cut, color='r')
    plt.axhline(y=bib_eff_cut, color='r')
    plt.tight_layout()
    if LAYER != -1 and LAYER != None:
        plt.savefig(f'roc-layer_{LAYER}.png')
    else:
        plt.savefig('roc.png')
    plt.clf()

    print(f"Cutoff at probability {prob_cutoff} yields {sig_eff_cut * 100}% Signal Efficiency \
    and {bib_eff_cut * 100}% BIB Efficiency")

    return prob_cutoff

def save_model(model, params, outputFilename=None, paramFilename=None):
    """
    Saves the model and param.
    Model's state_dict is saved, params are saved in a json.
    """
    if not outputFilename:
        outputFilename = "model.pt"
    if not paramFilename:
        paramFilename = "params.json"

    torch.save(model.state_dict(), outputFilename)
    print(f"Model successfully saved to: {outputFilename}.")
    with open(paramFilename, 'w') as json_file:
        json.dump(params, json_file)
    print(f"Params successfully saved to: {paramFilename}.")

def main(argv):
    if len(argv) == 0:
        noBibFilename, bibFilename, outputFilename, paramFilename = None, None, None, None
    elif len(argv) < len(["nobibfile, bibfile, outputfile, paramfile"]):
        print("usage: separation.py <nobibfile> <bibfile> <modelfile> <paramfile>")
        sys.exit(2)
    else:
        noBibFilename, bibFilename, outputFilename, paramFilename = argv

    layer_match = re.match(r".*layer_(\d+).root", noBibFilename)
    if layer_match:
        LAYER = int(layer_match.group(1))
    else:
        LAYER = -1

    data = InputData(bibFilename, noBibFilename)
    train_load, val_load, test_load, scaler = split_data(data)
    plot_all_features(data)

    # Initialize the NN and training parameters
    classifier = Net()
    optimizer = optim.SGD(classifier.parameters(), lr=0.05)
    #create class weights corresponding to bib and signal
    #0 is bib, 1 is signal
    class_weights = torch.FloatTensor([len(data) / (2.0 * data.num_bib), len(data) / (2.0 * data.num_sig)])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    training_losses, validation_losses = train_nn(classifier, optimizer, criterion, train_load, val_load)
    plot_loss(training_losses, validation_losses)

    scor0, scor1, label = check_test(classifier, test_load)
    plot_test_outputs(scor0, scor1, label)
    prob_cutoff = plot_best_cut(scor0, scor1, label)

    means = {f"mean_{i}" : scaler.mean_[i] for i in range(np.shape(scaler.mean_)[0])}
    scales = {f"scale_{i}" : scaler.scale_[i] for i in range(np.shape(scaler.scale_)[0])}
    params = {"cutoff": prob_cutoff}
    params.update(means)
    params.update(scales)
    save_model(classifier, params, outputFilename, paramFilename)

if __name__ == "__main__":
    main(sys.argv[1:])