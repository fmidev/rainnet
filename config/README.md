# Documentation of RainNet configuration

An example configuration is found in `example/`. This documentation aims to explain some non-trivial configuration intricacies to get training and inference working. It is to be used together with example configurations, that explain basics of each parameters.  

---

## FMI Dataset configuration
#### `FMIComposite.yaml`

All about reading in data for training and inference is defined here. You have two ways of reading composite data: directly from PGM files on disk (for example from `arch/radar/storage` on Athras) or from HDF5 files (Currently with `/data/PINCAST/manuscript_1/nowcasts/measurements/fmi_composite_raindays.hdf5` on Athras). HDF5 reading is recommended for Puhti (because of setting up easiness and reading speed there). The way is chosen using `importer : [pgm_gzip | hdf5]`.

- if using `pgm_gzip`
  - you have to specify `path` and `filename`, as specified in the example config.
  - `bbox` size must match `bbox_image_size` and `image_size` must match that of pgm files. 
- if using `hdf5`
  - you have to specify `hdf5_path` to point to the database. 
  - with `/data/PINCAST/manuscript_1/nowcasts/measurements/fmi_composite_raindays.hdf5`, `bbox_image_size`=`image_size=(512,512)`
- `input_image_size` a size divisible from `image_size` and compatible with the neural network architecture (with RainNet divisible by 32). 

---

## Model configuration
#### `rainnet.yaml`

training, prediction, and model configuration is defined here. 

- `val_batches` and `train_batches` given as integers give a precise number of batches to run on, whereas given as floats give a percentage of the respective dataset to use. 
- Which callback to use for early stopping, visualization, etc can be set from the `train_model.py` script. 
- Available choices for parameters like the loss function or the learning rate scheduler may be inspected from the model `LightningModule`.
- Further parameter structure for loss functions are found from respective classes in the `costfunctions` repository. 
- Predictions will be saved as HDF5 files in the directory structure format `{timestamp=last of input sequence}/{method=given_name}/{leadtime=1...L}/data` where `data` is the dataset where the prediction data resides. 
- model parameters are defined for the RainNet `nn.Module` defined in the `networks` repository. 


---

## Logging configuration
#### `output.yaml`

Simple logging configuration.

---

## Callback configurations
### Nowcasting metrics callback
#### `nowcast_metric_callback.yaml`

This callbacks adds metrics calculation and their Figures for the validation data. `ETS`, `CSI`, and `MAE` are implemented, but `CSI` is disabled in the code. 

- In `reduce_dims`: select all prediction data dimension indices except that for leadtime, if calculating metrics against leadtime. 
- Input mode must be set to `deterministic` if working with RainNet, L-CNN.

### Logging nowcast images callback
#### `log_nowcast_callback.yaml`

Callback for visualizing nowcasts made in the logger. 
