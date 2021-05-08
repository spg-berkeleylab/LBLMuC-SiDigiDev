# Creates a model for all the layers (separately)
from separation import main as create_model
import os

def main():
    num_layers = 4
    for i in range(num_layers):
        features_dir = os.path.join(os.getcwd(), "features")
        noBibFilename = os.path.join(features_dir, f"no-bib-features-layer_{i}.root")
        bibFilename = os.path.join(features_dir, f"bib-features-layer_{i}.root")
        model_dir = os.path.join(os.getcwd(), f"model-layer_{i}")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        outputFilename = os.path.join(model_dir, f"nn-layer_{i}.pt")
        paramFilename = os.path.join(model_dir, f"params-layer_{i}.json")
        create_model([noBibFilename, bibFilename, outputFilename, paramFilename])

if __name__ == "__main__":
    main()