## CLI Script for testing NN separation of BIB
## By Rohit Agarwal
## Given data and a separation model, ascertains how accurately the model predicts the data.
##-------------------------------------
## usage: py testing_separation.py <nobibfile> <bibfile> <modelfile> <paramfile>
## nobibfile - a ROOT file containing the features of a no-bib dataset
## bibfile - a ROOT file containing the features of a bib dataset
## modelfile - path to load the model. Should be a .pt file.
## paramfile - path to load the parameters. Should be a .json file.

import sep_classes as sep
import training_separation as train
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

def apply_params(data, paramFilename):
    """
    Applies standardization to a dataset given params in paramFilename stored as a json.
    The format must be of the output of training_separation.py.s
    Returns the params dictionary for future saving.
    """
    with open(paramFilename) as f:
        params = json.load(f)
    num_features = data[0][0].shape[0]
    scaler_mean = [params[f"mean_{i}"] for i in range(num_features)]
    scaler_scale = [params[f"scale_{i}"] for i in range(num_features)]
    transform = sep.StandardTransform(scaler_mean, scaler_scale)
    data.data = transform(data.data)
    return params

def plot_combined_roc(scor0, scor1, label):
    """
    Plots the ROC curve for all layers.
    """
    total = np.bincount(label)
    bib_eff = []
    sig_eff = []
    for cut in np.arange(0, 1, 0.01):
        results=np.bincount(label[cut<scor1], minlength=2)
        bib_eff.append(results[0] / total[0])
        sig_eff.append(results[1] / total[1])

    print(f"bib_eff = {bib_eff}")
    print(f"num_bib = {total[0]}")
    print(f"sig_eff = {sig_eff}")
    print(f"num_sig = {total[1]}")
    plt.plot(sig_eff, 1 - np.array(bib_eff))
    plt.title("BIB Rejection versus Signal Efficiency")
    plt.xlabel('Signal Efficiency')
    plt.ylabel('BIB Rejection')
    plt.xlim(1.1, 0)
    plt.tight_layout()
    plt.show()
    plt.savefig('roc.png')

def plot_feature(data, idx):
    plt.hist(data.data[data.labels==0,idx].numpy(),bins=100,label='0',density=True,histtype='step')
    plt.hist(data.data[data.labels==1,idx].numpy(),bins=100,label='1',density=True,histtype='step')
    plt.xlabel(f'Normalized feature {idx}')
    plt.title(f'Distribution of BIB vs non-BIB particles')
    plt.legend()
    plt.show()

def cut_feature(data, idx, thresh_low, thresh_high=float('inf')):
    """
    One dimensionally cuts on a specific (normalized) feature in the data.
    The number of the feature is given by idx.

    Returns: (bib_efficiency, signal_efficiency)
    """
    bib_data = data.data[data.labels==0,idx].numpy()
    sig_data = data.data[data.labels==1,idx].numpy()
    bib_kept = np.count_nonzero(bib_data < thresh_low)
    sig_kept = np.count_nonzero(sig_data < thresh_low)
    return bib_kept / len(bib_data), sig_kept / len(sig_data)

def main():
    parser = argparse.ArgumentParser(description="CLI Script for training NN separation of BIB")
    parser.add_argument('-m', "--minEff", dest="sig_eff_min", \
        help="Whether or not to include and save a new cutoff from a different minimum signal efficiency. (Default: Reads the cutoff from paramFile)")
    parser.add_argument('noBibFilename', nargs=1, help="A ROOT file containing the features of a no-bib dataset.")
    parser.add_argument('bibFilename', nargs=1, help="A ROOT file containing the features of a bib dataset.")
    parser.add_argument('modelFilename', nargs=1, help="Path to load the model. Should be a .pt file.")
    parser.add_argument('paramFilename', nargs=1, help="Path to load the parameters. Should be a .json file.")
    filenames = ["noBibFilename", "bibFilename", "modelFilename", "paramFilename"]
    args = vars(parser.parse_args())
    noBibFilename, bibFilename, modelFilename, paramFilename = tuple(args[name][0] for name in filenames)
    sig_eff_min = float(args["sig_eff_min"]) if args["sig_eff_min"] else None

    layer = None
    if noBibFilename:
        layer_match = re.match(r".*layer_(\d+).root", noBibFilename)
        if layer_match:
            layer = int(layer_match.group(1))
        else:
            layer = -1

    classifier = sep.Net()
    classifier.load_state_dict(torch.load(modelFilename))
    classifier.eval()

    data = sep.InputData(bibFilename, noBibFilename)
    params = apply_params(data, paramFilename)

    data_load = DataLoader(data, batch_size=512)

    train.plot_all_features(data, layer=layer)
    scor0, scor1, label = train.check_test(classifier, data_load)
    train.plot_test_outputs(scor0, scor1, label, layer=layer)
    # Can change this number to choose whatever cut you'd like
    if sig_eff_min:
        train.plot_best_cut(scor0, scor1, label, sig_eff_min=sig_eff_min, layer=layer)
    else:
        train.plot_best_cut(scor0, scor1, label, prob_cutoff=params["cutoff"], layer=layer)


if __name__ == "__main__":
    main()