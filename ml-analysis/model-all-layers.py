# Creates a model for all the layers (separately)
from separation import main as create_model
import os

def main():
    num_layers = 4
    for i in range(num_layers):
        bibFilename = os.path.join("features", f"bib-features-layer_{i}")
        noBibFilename = os.path.join("features", f"no-bib-features-layer_{i}")
        outputFilename = os.path.join(f"model-layer_{i}", f"nn-layer_{i}")
        paramFilename = os.path.join(f"mode-layer_{i}", f"params-layer_{i}")
        create_model([bibFilename, noBibFilename, outputFilename, paramFilename])

if __name__ == "__main__":
    main()