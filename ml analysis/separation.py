# %%
import numpy as np
import matplotlib.pyplot as plt
import uproot as ur
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%matplotlib inline
# %%
nobibFile = ur.open("./runs/10GeV-no-bib/ntuple_tracker.root")
nobibTuple = nobibFile["MyLCTuple"]
bibFile = ur.open("./runs/10GeV-bib/ntuple_tracker.root")
bibTuple = bibFile["MyLCTuple"]
# %%
def createFeaturesMatrix(ttree):
    """
    Creates an n x 4 ndarray where n is the amount of clusters in the ttree.
    The ndarray has data along its rows.
    """
    # For training purposes, scale of charge doesn't matter
    featureBranches = ['thpox', 'thpoy', 'thpoz', 'thedp']
    rawFeatures = len(featureBranches)
    mapFeatures = list(map(lambda a: np.concatenate(a), ttree.arrays(expressions=featureBranches, library='np').values()))
    # Feature branches + 2 size features + 1 theta feature. Can be adjusted for more features
    features = np.empty((rawFeatures + 2 + 1, len(mapFeatures[0])))
    features[:rawFeatures,:] = np.array(mapFeatures)
    
    #Cluster size features computed separately
    ntrhs = ttree["ntrh"].array(library='np')
    thcidx = ttree["thcidx"].array(library='np')
    thclen = ttree["thclen"].array(library='np')
    tcrp0 = ttree["tcrp0"].array(library='np')
    tcrp1 = ttree["tcrp1"].array(library='np')

    total_index = 0
    for (ie, ntrh) in enumerate(ntrhs):
        for ic in range(ntrh):
            idx = thcidx[ie][ic]
            ln = thclen[ie][ic]
            minX = min(tcrp0[ie][idx:idx + ln])
            maxX = max(tcrp0[ie][idx:idx + ln])
            minY = min(tcrp0[ie][idx:idx + ln])
            maxY = max(tcrp0[ie][idx:idx + ln])
            features[rawFeatures:rawFeatures+2, total_index] = np.array([maxX - minX + 1,maxY - minY + 1])
            total_index += 1

    #Theta branch. We can also add r later
    poy = features[1, :] 
    poz = features[2, :]
    np.arctan(poy * (poz ** -1), out=features[-1, :])
    np.add(features[-1, :], np.pi, out=features[-1, :], where=features[-1, :]<0)
    # Calculate abs(theta - 90 deg)
    # np.add(features[-1, :], - np.pi / 2, out=features[-1, :])
    # np.abs(features[-1, :], out=features[-1, :])
    #pox, poy, poz didn't help data
    features = features[3:,:]
    # Theta, X,Y size, Energy
    features = np.transpose(features)
    np.random.shuffle(features)
    return features
# %%
single_mu_data = createFeaturesMatrix(nobibTuple)
bib_data = createFeaturesMatrix(bibTuple)
# %%
#Plot y cluster size vs. theta
fig = plt.figure(figsize = (20,5))
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(10,5))
ax1.set(ylim = (0, 45))
ax2.set(ylim = (0, 45))
ax1.scatter(bib_data[:5000,-1],bib_data[:5000,-2], s=1)
ax1.set_ylabel("Y cluster size")
ax1.set_xlabel("Theta of cluster")
ax2.scatter(single_mu_data[:5000,-1],single_mu_data[:5000,-2], s=1)
ax2.set_ylabel("Y cluster size")
ax2.set_xlabel("Theta of cluster")
plt.suptitle("Y cluster size vs Theta (Left BIB, Right single mu)")
plt.tight_layout()
# %%
training_ratio = 0.6
#Everything in [0, split_mu) used for training
#This can also serve as an indicator between a point being bib or not.
split_mu = round(training_ratio * np.shape(single_mu_data)[0])
#correct for bib being much more data than non-bib
split_bib = split_mu
single_training = single_mu_data[:split_mu,:]
bib_training = bib_data[:split_bib,:]
training_data = np.vstack([single_training, bib_training])
training_data = StandardScaler().fit_transform(training_data)
# %%
PCA_data = PCA(n_components=2).fit_transform(training_data)
# %%
single_mu_pca = PCA_data[:split_mu, :]
bib_pca = PCA_data[split_mu:, :]
plt.figure(figsize = (10, 5))
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
ax1.set(xlim = (-1, 2))
ax2.set(xlim = (-1, 2))
plt.suptitle("Clusters in PCA basis")
ax1.scatter(single_mu_pca[:,0], single_mu_pca[:,1], s=1, c=['red'])
ax1.legend(['Single Mu'])
ax2.scatter(bib_pca[:, 0], bib_pca[:, 1], s=1, c=['green'])
ax2.legend(['BIB'])
ax1.set_xlabel("Projection onto 1st P.C.")
ax2.set_xlabel("Projection onto 1st P.C.")
ax1.set_ylabel("Projection onto 2nd P.C.")
ax2.set_ylabel("Projection onto 2nd P.C.")
plt.tight_layout()
# %%
CUT = 0.0
print(f"Single Mu Cut Efficiency (Training): {np.count_nonzero(PCA_data[:split_mu, 0] < CUT) / split_mu}")
print(f"BIB Cut Efficiency (Training): {np.count_nonzero(PCA_data[split_mu:, 0] < CUT) / split_bib}")
# %%
# %%
# First two layers only
# 1. Improving separation
# -- Simple cuts
# -- Pushing with multivariate technique (correlation between variables)
# 2. Removing separation bias
# plot of efficiency vs theta
# What is loss?
# %%
