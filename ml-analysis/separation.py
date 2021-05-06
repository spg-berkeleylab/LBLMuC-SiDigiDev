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



# %%
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

# %%
data = InputData()

# %%
# ratio of training : total
training_ratio = 0.5
# ratio of validation : total
val_ratio = 0.25

len_train, len_val = round(0.5 * len(data)), round(0.25 * len(data))
len_test = len(data) - len_train - len_val

# indices inhabited by each set
train_ind, val_ind, test_ind = random_split(np.arange(len(data)), [len_train, len_val, len_test])

train_ind, val_ind, test_ind = list(train_ind), list(val_ind), list(test_ind)
train_set = TransformableSubset(data, train_ind)
print('train_data:', data.data[train_ind])
# fit the transform to the train_set, but transform all the data
scaler = train_set.fit()
print('mean:', scaler.mean_)
print('scale:', scaler.scale_)
transform = StandardTransform(scaler.mean_, scaler.scale_)
print(f"Data before normalization: {data.data}")
data.data = transform(data.data)
print(f"Data after normalization: {data.data}")

# %%
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
#Create samplers to randomly sample the data
train_sampler = SubsetRandomSampler(train_ind)
val_sampler   = SubsetRandomSampler(val_ind)
test_sampler  = SubsetRandomSampler(test_ind)
#Use data-loaders to create iterables
train_load = DataLoader(data, batch_size=512, sampler=train_sampler)
val_load   = DataLoader(data, batch_size=256, sampler=val_sampler  )
test_load  = DataLoader(data, batch_size=256, sampler=test_sampler )
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
optimizer = optim.SGD(classifier.parameters(), lr=0.05)
#create class weights corresponding to bib and signal
#0 is bib, 1 is signal
class_weights = torch.FloatTensor([len(data) / (2.0 * data.num_bib), len(data) / (2.0 * data.num_sig)])
criterion = nn.CrossEntropyLoss(weight=class_weights)
# %%

training_losses   = []
validation_losses = []
# training loop
MAX_EPOCHS = 20

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

# %%
#Plot training loss
plt.plot(training_losses  ,label='Training')
plt.plot(validation_losses,label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('loss.png')
plt.show()
plt.clf()

# %%
false_signal = []
true_signal = []
signal_kept, total_signal, bib_kept, total_bib = 0, 0, 0, 0
for samples_batched, labels_batched in test_load:
    output = F.softmax(classifier(samples_batched), dim=-1)
    pred_classes = torch.argmax(output, dim=1)
    true_labels = labels_batched
    actual_signal = (true_labels == 1)
    actual_bib = (true_labels == 0)
    kept = (pred_classes == 1)

    false_signal.extend([output[i][1].item() for i in range(len(output)) if true_labels[i] == 0])
    true_signal .extend([output[i][1].item() for i in range(len(output)) if true_labels[i] == 1])

    total_signal += torch.count_nonzero(actual_signal).item()
    signal_kept  += torch.count_nonzero(np.logical_and(actual_signal, kept)).item()
    total_bib    += torch.count_nonzero(actual_bib   ).item()
    bib_kept     += torch.count_nonzero(np.logical_and(actual_bib   , kept)).item()

signal_eff = signal_kept / total_signal
bib_eff = bib_kept / total_bib
print(f"Cut Efficiency (Signal): {signal_eff}")
print(f"Cut Efficiency (BIB): {bib_eff}")
# %%
# Plot the probability that the classifier thinks each test sample is signal.
plt.hist(false_signal,label='BIB'   ,histtype='step',density=True)
plt.hist(true_signal ,label='Signal',histtype='step',density=True)
plt.axvline(x=0.5, color='r')
plt.xlabel("Probability of Signal")
plt.tight_layout()
plt.legend()
plt.savefig('prob.png')
plt.show()
plt.clf()