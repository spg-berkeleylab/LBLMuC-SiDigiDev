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
#create training data: TODO use torch's dataset classes
training_ratio = 0.5
split_signal = int(np.ceil(training_ratio * np.shape(signal_data)[0]))
split_bib = int(np.ceil(training_ratio * np.shape(bib_data)[0]))
signal_training = signal_data[:split_signal,:]
bib_training = bib_data[:split_bib,:]
#signal is 1s, bib is 0s
labels = torch.tensor((np.hstack([np.ones(split_signal), np.zeros(split_bib)]))).long()
training_data = torch.tensor(np.vstack([signal_training, bib_training])).float()
#shuffle the data and labels with the same seed
seed = np.random.randint(0, 2**(16))
np.random.seed(seed)
np.random.shuffle(training_data)
np.random.seed(seed)
np.random.shuffle(labels)

#normalize data (is this needed?)
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

# %%
#create validation data
valid_ratio = 0.5
val_split_ratio = training_ratio + (1 - training_ratio) * valid_ratio
val_signal = int(np.ceil(val_split_ratio * np.shape(signal_data)[0]))
val_bib = int(np.ceil(val_split_ratio * np.shape(bib_data)[0]))
signal_val = signal_data[split_signal:val_signal,:]
bib_val = bib_data[split_bib:val_bib,:]
val_signal_n = np.shape(signal_test)[0]
val_bib_n = np.shape(bib_test)[0]
labels_val = torch.tensor((np.hstack([np.ones(val_signal_n), np.zeros(val_bib_n)]))).long()
val_data = torch.tensor(np.vstack([signal_val, bib_val])).float()
seed = np.random.randint(0, 2**(16))
np.random.seed(seed)
np.random.shuffle(val_data)
np.random.seed(seed)
np.random.shuffle(labels_val)

#normalize data with the same scaling as the training data.
val_data = scaler.transform(val_data)

# %%
#create test data: 
signal_test = signal_data[val_signal:,:]
bib_test = bib_data[val_bib:,:]
test_signal_n = np.shape(signal_test)[0]
test_bib_n = np.shape(bib_test)[0]
labels_test = torch.tensor((np.hstack([np.ones(test_signal_n), np.zeros(test_bib_n)]))).long()
test_data = torch.tensor(np.vstack([signal_test, bib_test])).float()
seed = np.random.randint(0, 2**(16))
np.random.seed(seed)
np.random.shuffle(test_data)
np.random.seed(seed)
np.random.shuffle(labels_test)

#normalize data with the same scaling as the training data.
test_data = scaler.transform(test_data)
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

last_ind = test_signal_n + test_bib_n
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
plt.title("Prediction of signal")
plt.hist(bib_hot)
plt.axvline(x=0.5, color='r')
plt.xlabel("Probability of being signal")
plt.ylabel("Number at that probability")
plt.show()