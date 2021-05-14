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
import sys
import argparse
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
    sig_eff_min = float(args["sig_eff_min"])

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