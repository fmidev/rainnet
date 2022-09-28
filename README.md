# RainNet-PyTorch
PyTorch implementation of RainNet (Ayzel et al. 2020)

[![DOI](https://zenodo.org/badge/541990111.svg)](https://zenodo.org/badge/latestdoi/541990111)

Added features include:

- Training using multiple prediction leadtimes
- A wider range of loss functions usable: e.g. MS-SSIM, Gaussian NLL loss...

An instance of the configuration that has to be set up for training RainNet and performing inference is defined in a folder under `config`, consisting of a collection of `YAML` files. Documentation of the configuration is found at `config/README.md` and an example configuration is available in `config/example`.

- run `python train_model.py [CONFIG FOLDER INSIDE config/] -c [CHECKPOINT PATH]` for training the model. Checkpoint is facultative.
- run `python predict_model.py  [CHECKPOINT PATH] [CONFIG FOLDER INSIDE config/]` for running and saving predictions for a trained model.
- `predict_model_pysteps.py` is an alternative (unmaintained and deprecated) script for running and saving predictions that uses PYSTEPS for IO of composites but doesn't use GPU for inference.
