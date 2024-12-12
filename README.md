# High Precision Regressors for Particle Physics simulations (qqzz4l-NN)

> Neural Network implementations of surrogate regressors for qqZZ4l.


## About

This repository contains the code for the high-precision regressors that can be trained on particle physics simulation data. 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## Data Description

The dataset `qqzz4l-NN.data.tar.gz` contains the train, eval and test datasets. 
They have the following fields:

<!-- Phase space coordinates : $x_1$, $x_2$, $x_3$, $x_4$  -->
The corresponding helicity amplitudes $h_1$, $h_2$, . . . , $h_9$
Each helicity amplitude has two classes [A+B] and [C]. 
For each class there are real and imaginary parts.
The real and imaginary components of the helicity amplitudes are present in this order as $y_1, \dots, y_{36}$ in the dataset.

The following is the mapping of the helicity amplitudes to the y columns of the dataset

| **Description**                                          | **h1** | **h2** | **h3** | **h4** | **h5** | **h6** | **h7** | **h8** | **h9** |
|----------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| the real part of the classes A+B 2-loop amplitude         | y1     | y5     | y9     | y13    | y17    | y21    | y25    | y29    | y33    |
| the imaginary part of the classes A+B                    | y2     | y6     | y10    | y14    | y18    | y22    | y26    | y30    | y34    |
| the real part of the class C 2-loop amplitudes            | y3     | y7     | y11    | y15    | y19    | y23    | y27    | y31    | y35    |
| the imaginary part of the class C                        | y4     | y8     | y12    | y16    | y20    | y24    | y28    | y32    | y36    |

For testing and validation we can use an alternative set of data termed as `Monte Carlo` or `MC` data.

## The prediction task
Given the phase space coordinates (x-values) we predict the helicity amplitude components (y-values). 

We can specify the y-values that are needed to be predicted during the training. For example: `y1, y2, y5, y6`

We can also specify a prediction on an arithmetic expression between two y-values. 
For example: `y2+y3`

We can also specify a mix of y-values and expressions with y-values. For example: `y1, y2, y2+y3, y5-y6`

We currently support the arithmetic operations between 2 y-values using the respective operators + , - , * . Division operation is supported with the / operator. However, caution needs to be exercised when there are zero-values.


## Directory structure
```
QQZZ4L-NN/
├── checkpoints/         # contains the model training checkpoints
├── data/                # contains the script to download data
├── models/              # contains the model weights, test results, and distribution plots and evaluation reports (PDF).
├── notebooks/           # contains the notebooks for drawing plots, and debugging
├── scripts/             # contains the code for training the models (python code and batch scripts)
```

## The Config (JSON) file
The config file is used extensively in this repository to keep track of the model's training configuration. The user needs to specify a JSON file that takes different arguments for the training program to run.

The JSON has these fields

*   **model_type**: Defines the type of model. Can be any of: `dnn, skip, skip-stream, skip-light`.
*   **input_shape**: Specifies the shape of the input data. Default: `4`.
*   **amplitude_shape**: Defines the shape of the helicity-amplitude data. Default: `36`.
*   **data_dir**: Directory where the dataset is located. Example: `"/scratch/username/dataset"`.
*   **seed**: Seed value for reproducibility. Example: `42`.
*   **var_y**: List of helicity-amplitude variable combinations. Example: `["y1", "y5", "y5+y13", "y6+y14"]`.
*   **activation**: Activation function used in the model. Can be any of: `'relu', 'elu', 'swish' (silu), 'leaky_relu', 'softplus'`.
*   **width**: The width of the skip layers. Example: `32`.
*   **depth**: Number of skip modules in the model. Example: `16`.
*   **skip_block_layers**: Number of layers in the skip module. Example: `3`.
*   **beta**: the weight of the L2 regularization. Example: `0`.
*   **alpha**: the weight of the L1 regulatization. Example: `0`.
*   **normal_scaled**: Boolean to determine whether the scaled data is normalized. Example: `false`.
*   **lr_decay_type**: Learning rate decay strategy. Can be any of: `exp, poly, const`. (denoting exponential, polynomial and constant decay respectively)
*   **initial_lr**: Initial learning rate. Example: `0.003`.
*   **final_lr**: Final learning rate after decay. Example: `1e-06`.
*   **decay_steps**: Number of steps involved in decaying the learning rate. Example: `120000`.
*   **train-sample-size**: The number of samples used for training. Example: `30000000`.
*   **validate-sample-size**: Number of validation samples. Example: `500000`.
*   **test-sample-size**: Number of test samples. Example: `500000`.
*   **use_MC_sample**: Indicates if Monte Carlo data is used for the test and validation. Example: `false`.
*   **batch_size**: Number of samples per gradient update. Example: `1024`.
*   **steps_per_epoch**: Number of steps taken per epoch. Example: `2000`.
*   **early_stopping_start_epoch**: Epoch number after which early stopping starts being monitored. Example: `50`.
*   **patience**: Number of epochs with no improvement after which training will stop. Example: `150`.
*   **monitor**: The metric used for monitoring the training process. Example: `"val_mse"`.
*   **loss**: The loss function to be minimized. Example: `"mse"`.
*   **gradient_clipping**: Boolean to indicate if gradient clipping is used. Example: `true`.
*   **verbose**: Verbosity mode. Example: `1`.
*   **base_directory**: Directory to save the models. It is relative to the directory where the program is being run. Example: `"../models/"`.
*   **epochs**: Total number of epochs to train the model. Example: `1000`.
*   **model-uuid**: Unique identifier for the model. Default: `"UUID"`. (generates a random UUID). If you want to give a specific model id, it can be specified here.
*   **checkpoint_path**: Load the model training from a particular checkpoint. Give the name of the checkpoint directory present in the checkpoints folder. 

Sample config JSON can be found in the `scripts` folder (in the root directory) with the name `config-modelX.json` where `X` is an arbitrary number. 

## Functionalities supported
1. Training of the regressor models
1. Automatic Hyperparameter Tuning and Neural Architecture Search to find the model config
1. Visualizing Training and Testing results, and Parameter tuning results

### What happens when the training is run

When a training is run, it creates a folder with its uuid in the `models` folder and the `checkpoints` folder. If an sbatch job is submitted, a corresponding folder is created inside the `scripts/batch_scripts/run_logs` folder.

### What happens when the hyperparameter tuning is run

When a hyperparameter tuning is run, it creates a file in the `models` folder which contains the `study` object from the `Optuna` library that has the results of the parameter tuning study. This can be visualized using a notebook in the notebooks folder.

## Description of the folders

### The models folder
A particular model folder in the `models` directory contains the training configuration, train history, and trained model files (JSON, Torch), test results (CSV), and distribution plots and evaluation reports (PDF).

### The checkpoints folder
The checkpoints folder contains folders corresponding to the models trained (named after the model uuid). A particular model folder contains the most recent training checkpoint information.

### Scripts folder
The scripts folder contains the code of this project. It contains the code required to load the data, create and train the models, tune the hyperparameters, and visualize the results. It has some files and sub-folders within it.

### Notebooks folder
This folder contains different notebooks for helping to visualize the results and some notebooks created for debugging purpose.

`torch-NN.py` contains the code used to load the dataset, train, and test the model

`torch-MCNN.py` contains the code used to load the dataset, train, and test the model but for an alternative model (Multi-Column Neural Network implementation)

`supporting` directory contains the code from `torch-NN.py` that is reused as helper functions for some other code files to be used.

`autotune-torch-NN.py` contains the code for automatic Hyperparameter Tuning and Neural Architecture Search

`config.json` the files with the name config and the extension json in it will have the configuration for training the models

`noprune-autotune-torch-NN.py` an experimental file created to perform hyperparameter tuning with Optuna without pruning trials

`vloss-autotune-torch-NN.py` an experimental file created to perform hyperparameter tuning with Optuna using the best validation loss in the trial as the optimization metric

#### The batch_scripts and the run_logs folder
The `batch_scripts` folder under `scripts` contains the sbatch shell script files that will run the training in the background on a cluster. Some sample batch scripts are included with the name `model_X.sh` where `X` is some number.

The `run_logs` folder inside the `batch_scripts` will store the log of execution of a batch_script whenever it is run.

## Installation
```bash
# create a conda environment
conda create --name myenv

# activate the conda environment
conda activate myenv

# Install dependencies
pip install -r requirements.txt

# OPTIONAL: For execution in jupyterlab notebook, create an ipykernel
pip install ipykernel

python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

# ensure the data is extracted in the folder which is specified in the json
tar -xzf qqzz4l-NN.data.tar.gz

```

## Suggested environment
The model training has been tested on the Discovery cluster with a V100 GPU. The training configuration can be found in the batch script files in the `scripts/batch_scripts` folder. 


## Instructions for training (live)
1. Navigate to `/scripts` folder.
1. Activate the conda environment.
1. Ensure your model config file is present in the `/scripts` directory.
1. Ensure that the dataset path is updated in the config json file before training.
1. Use the command `python torch-NN.py config.json` where `config.json` is the training config file, and the path is relative to `/scripts` (as we  currently navigated to that directory)

## Instructions for training (background / batch script)
1. Navigate to `/scripts` folder.
1. Ensure your model config file is present in the `/scripts` directory.
1. Ensure that the dataset path is updated in the config json file before training.
1. Navigate to the `batch_scripts` folder inside `scripts`. 
1. Ensure the shell script to run the batch job is present inside the `batch_scripts` folder.
1. Navigate back to the `/scripts` folder.
1. Use the command `sbatch batch_scripts/model.sh` where `model.sh` is the shell script to run the batch job.

Sample shell scripts for the background job submission can be found in the `scripts/batch_scripts` folder with the name `model_X.sh`

### Loading the model from a checkpoint for training
If the training terminates after running for some number of epochs or if the batch job has passed the time-limit, we can resume the training from the last saved checkpoint.

To do so, set the `"checkpoint_path"` argument of the config json to the model uuid of the previously trained model. Before that, ensure the checkpoints directory contains an entry by the name of the previous model.

## Instructions for automatic hyperparameter tuning (live)
1. Navigate to `/scripts` folder.
1. Activate the conda environment.
1. Ensure that the dataset path is updated in the `autotune-torch-NN.py` file before training.
1. Adjust the range of the parameters to be tuned as necessary.
1. Use the command `python autotune-torch-NN.py` to start the tuning process.

## Instructions for automatic hyperparameter tuning (background / batch script)
1. Navigate to `/scripts` folder.
1. Activate the conda environment.
1. Ensure that the dataset path is updated in the `autotune-torch-NN.py` file before training.
1. Adjust the range of the parameters to be tuned as necessary.
1. Navigate to the `batch_scripts` folder inside `scripts`. 
1. Ensure the shell script to run the batch job is present inside the `batch_scripts` folder.
1. Navigate back to the `/scripts` folder.
1. Use the command `sbatch batch_scripts/model_tune.sh` where `model_tune.sh` is the shell script to run the batch job.

Example shell script for the background job submission can be found in the `scripts/batch_scripts` folder with the name `model_tune.sh`

## Visualizations

The visualizations for the training and testing can be checked after the model has finished the training process. The notebook to run visualizations is present in `notebooks` folder. 
- The code to make the plots is present in the `plots-new.ipynb` notebook.
- To draw plots for a trained model, navigate to the `models` directory corresponding to a particular model.
- Locate the test-results file for the model and copy its path. The path should be relative to the notebook's location.
- Paste the path in one of the sections of the notebook that loads the dataframe.
- Execute the rest of the histogram plotting cells to obtain the visualizations.

Similarly, the visualizations for the hyperparameter tuning can be checked by accessing the notebook `hyperparameter_tuning_results.ipynb`. The default path for the `study` object has already been set in the notebook, however, this can be changed as per convenience. 