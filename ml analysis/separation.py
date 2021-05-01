# %%
import numpy as np
import matplotlib.pyplot as plt
import uproot as ur
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
#%matplotlib inline
# %%
nobibFile = ur.open("no-bib-features.root")
nobibTuple = nobibFile["FeaturesTree"]
bibFile = ur.open("bib-features.root")
bibTuple = bibFile["FeaturesTree"]
# %%
def createFeaturesMatrix(ttree):
    """
    Creates an n x 5 ndarray where n is the amount of clusters in the ttree.
    The ndarray has data along its rows.
    """
    featureBranches = ['clszx', 'clszy', 'clch', 'clthe', 'clpor']
    features = np.transpose(np.array(list(ttree.arrays(expressions=featureBranches, library='np').values())))
    return features
# %%
signal_data = createFeaturesMatrix(nobibTuple)
bib_data = createFeaturesMatrix(bibTuple)

# %%
#Plot y cluster size vs. theta as a sanity check
fig = plt.figure(figsize = (20,5))
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(10,5))
ax1.set(ylim = (0, 45))
ax2.set(ylim = (0, 45))
ax1.scatter(bib_data[:5000,3],bib_data[:5000,1], s=1)
ax1.set_ylabel("Y cluster size")
ax1.set_xlabel("Theta of cluster")
ax2.scatter(signal_data[:5000,3],signal_data[:5000,1], s=1)
ax2.set_ylabel("Y cluster size")
ax2.set_xlabel("Theta of cluster")
plt.suptitle("Y cluster size vs Theta (Left BIB, Right signal)")
plt.tight_layout()
# %%
training_ratio = 0.6
split_signal = round(training_ratio * np.shape(signal_data)[0])
split_bib = round(training_ratio * np.shape(bib_data)[0])
signal_training = signal_data[:split_signal,:]
bib_training = bib_data[:split_bib,:]
# non_bib weighted by frequency, bib kept at 0

labels = torch.FloatTensor((np.hstack([np.ones(split_signal), np.zeros(split_bib)]))).unsqueeze(-1)
training_data = np.vstack([signal_training, bib_training])
#shuffle the data and labels with the same seed
seed = np.random.randint(0, 2**(16))
np.random.seed(seed)
np.random.shuffle(training_data)
np.random.seed(seed)
np.random.shuffle(labels)

#normalize data (is this needed?)
training_data = MinMaxScaler().fit_transform(training_data)
# %%
class Net(nn.Module):
    """
    Neural network to compute classification for BIB vs signal. 
    4-layer neural network
    5 - 5 - 3 - 1
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    #backprop is automatically made by PyTorch
# %%
classifier = Net()
optimizer = optim.SGD(classifier.parameters(), lr=0.01)
#create class weights corresponding to bib and signal
total_data = split_signal + split_bib
class_weights = torch.FloatTensor([total_data / (2.0 * split_signal), total_data / (2.0 * split_bib)])
criterion = nn.BCELoss(weight=class_weights)

losses = []
def run_training_step(batch_begin, batch_end):
    """
    Runs the training loop on the row in the indices between the two arguments:
    [batch_begin, batch_end)
    """
    optimizer.zero_grad()   # zero the gradient buffers
    batch = torch.FloatTensor(training_data[batch_begin:batch_end,:])
    output = classifier(batch)
    target = labels[batch_begin:batch_end]
    print(f"Shape of output: {np.shape(output)}")
    print(f"Shape of target: {np.shape(target)}")
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# training loop
BATCH_SIZE = 32
                #total_data
for it in range(0, 32, BATCH_SIZE):
    run_training_step(it, it + BATCH_SIZE)

if it < total_data:
    run_training_step(it, total_data)
# %%
# First two layers only
# 1. Improving separation
# -- Simple cuts
# -- Pushing with multivariate technique (correlation between variables)
# 2. Removing separation bias
# plot of efficiency vs theta
# What is loss?
# %%
