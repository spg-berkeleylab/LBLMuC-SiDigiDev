# %%
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

# %%
class InputData(Dataset):
    """
    Creates a labeled dataset from the bibFilename and noBibFilename root files specified.
    Optionally include a transform to be applied on the data.
    """
    def __init__(self, bibFilename="bib-features.root", nobibFilename="no-bib-features.root"):
        bibFile = ur.open(bibFilename)
        bibTuple = bibFile["FeaturesTree"]
        nobibFile = ur.open(nobibFilename)
        nobibTuple = nobibFile["FeaturesTree"]
        
        bib_data = InputData.createFeaturesMatrix(bibTuple)
        signal_data = InputData.createFeaturesMatrix(nobibTuple)
        self.data = torch.vstack([signal_data, bib_data])
        self.labels = torch.hstack([torch.ones(signal_data.shape[0]), \
                                    torch.zeros(bib_data.shape[0])])

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

# %%
class TransformableSubset(InputData):
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
        return (sample - self.mean)/self.scale

# %%
data = InputData()
# ratio of training : total
training_ratio = 0.5
# ratio of validation : total
val_ratio = 0.25
len_train, len_val = round(0.5 * len(data)), round(0.25 * len(data))
len_test = len(data) - len_train - len_val
# indices inhabited by each set
train_ind, val_ind, test_ind = random_split(np.arange(len(data)), [len_train, len_val, len_test])
train_set = TransformableSubset(data, train_ind)
# fit the transform to the train_set, but transform all the data
scaler = train_set.fit()
transform = StandardTransform(scaler.mean_, scaler.scale_)
print(f"Data before normalization: {data.data}")
data.data = transform(data.data)
print(f"Data after normalization: {data.data}")

# %%
#Create samplers to randomly sample the data
train_sampler = SubsetRandomSampler(data, train_ind)
val_sampler = SubsetRandomSampler(data, val_ind)
test_sampler = SubsetRandomSampler(data, test_ind)
#Use data-loaders to create iterables
train_load = DataLoader(data, batch_size=32, sampler=train_sampler)
val_load = DataLoader(data, batch_size=16, sampler=val_sampler)
test_load = DataLoader(data, batch_size=16, sampler=test_sampler)
# %%
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
classifier = Net()
optimizer = optim.SGD(classifier.parameters(), lr=0.01)
#create class weights corresponding to bib and signal
#0 is bib, 1 is signal
total_data = split_signal + split_bib
class_weights = torch.FloatTensor([total_data / (2.0 * split_bib), total_data / (2.0 * split_signal)])
criterion = nn.CrossEntropyLoss(weight=class_weights)

training_losses = []
def run_training_step(batch_begin, batch_end):
    """
    Runs the training loop on the row in the indices between the two arguments:
    [batch_begin, batch_end)
    """
    optimizer.zero_grad()   # zero the gradient buffers
    batch = torch.tensor(training_data[batch_begin:batch_end]).float()
    output = classifier(batch)
    target = labels[batch_begin:batch_end]
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())

validation_losses = []
def run_validation_step(batch_begin, batch_end):
    """
    Runs a validation loop on the row in the indices between the two arguments:
    [batch_begin, batch_end)
    """
    batch = torch.tensor(val_data[batch_begin:batch_end]).float()
    output = classifier(batch)
    target = labels_val[batch_begin:batch_end]
    loss = criterion(output, target)
    validation_losses.append(loss.item())
# %%
# training loop
TRAINING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 16
EPOCHS = 1200 #2000 epochs seemed to make it converge
for epoch in range(0, EPOCHS):
    training_ind = epoch * TRAINING_BATCH_SIZE
    run_training_step(training_ind, training_ind + TRAINING_BATCH_SIZE)
    valid_ind = epoch * VALIDATION_BATCH_SIZE
    run_validation_step(valid_ind, valid_ind + VALIDATION_BATCH_SIZE)

#if using all of the data, could have extra, smaller batch at the end
# %%
#Plot training loss
plt.plot(range(len(training_losses)), training_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# %%
#Plot validation loss
plt.plot(range(len(validation_losses)), validation_losses)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# %%
bib_hot = []
def run_testing_step(batch_begin, batch_end):
    """
    Runs the testing step on the row in the indices between the two arguments:
    [batch_begin, batch_end)
    Returns tuple: (# of signal kept, # of total signal, # of BIB kept, # of total BIB) in the batch
    """
    batch = torch.tensor(test_data[batch_begin:batch_end]).float()
    output = F.softmax(classifier(batch), dim=-1)
    pred_classes = torch.argmax(output, dim=1)
    true_labels = labels_test[batch_begin:batch_end]
    actual_signal = (true_labels == 1)
    actual_bib = (true_labels == 0)
    kept = (pred_classes == 1)
    bib_hot.extend([p[1].item() for p in output])

    total_signal = torch.count_nonzero(actual_signal).item()
    signal_kept = torch.count_nonzero(actual_signal & kept).item()
    total_bib = torch.count_nonzero(actual_bib).item()
    bib_kept = torch.count_nonzero(actual_bib & kept).item()
    return signal_kept, total_signal, bib_kept, total_bib

#last_ind = test_signal_n + test_bib_n
last_ind = total_data
total_signal_kept, total_overall_signal, total_bib_kept, total_overall_bib = 0, 0, 0, 0
for it in range(0, last_ind, BATCH_SIZE):
    signal_kept, total_signal, bib_kept, total_bib = run_testing_step(it, it + BATCH_SIZE)
    total_signal_kept += signal_kept
    total_bib_kept += bib_kept
    total_overall_signal += total_signal
    total_overall_bib += total_bib

signal_eff = total_signal_kept / total_overall_signal
bib_eff = total_bib_kept / total_overall_bib
print(f"Cut Efficiency (Signal): {signal_eff}")
print(f"Cut Efficiency (BIB): {bib_eff}")
# %%
# Plot the probability that the classifier thinks each test sample is signal.
plt.title("Prediction of signal from BIB")
plt.hist([bib_hot[i] for i in range(len(bib_hot)) if labels[i] == 0])
# plt.axvline(x=0.5, color='r')
plt.xlabel("Probability of BIB being signal")
plt.ylabel("Number at that probability")
plt.show()


# %%
plt.title("Prediction of signal from signal")
plt.hist([bib_hot[i] for i in range(len(bib_hot)) if labels[i] == 1])
# plt.axvline(x=0.5, color='r')
plt.xlabel("Probability of signal being signal")
plt.ylabel("Number at that probability")
plt.show()

