#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import sys
import logging
import uuid
import os
import shutil
import json
import argparse
import pytorch_model_summary as pms
import pickle


from matplotlib import rc
from supporting.load_data import load_data, build_data_loaders, x_scale, y_scale, y_unscale
from supporting.nn_models import get_device, EarlyStopping, skip_block, getActivation, skip_dnn, dnn, skip_light_module, skip_light, nets
import optuna
from optuna.trial import TrialState

rc('text', usetex=False)
# plt.rcParams['text.latex.preamble'] = []
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams['font.weight'] = 'light'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CLIP = 1e12

def get_config():
    config_template = {
        "model_type": "skip-light",
        "input_shape": 4,
        "amplitude_shape": 36,
        "data_dir": "/scratch/akula.ha/dataset", 
        "seed": 42,
        "var_y": ["y5", "y6"],
        "activation": "leaky_relu", 
        "width": 32, 
        "depth": 10, 
        "skip_block_layers": 5, 
        "beta": 0,
        "alpha": 0,
        "normal_scaled": False,
        "lr_decay_type": "exp",
        "initial_lr": 0.003,
        "final_lr": 1e-06,
        "decay_steps": 120000, 
        "train-sample-size": 10000000,
        "validate-sample-size": 500000,
        "test-sample-size": 500000,
        "use_MC_sample": False,
        "batch_size": 1024,
        "steps_per_epoch": 900,
        "early_stopping_start_epoch": 50, 
        "patience": 50,
        "monitor": "val_mse",
        "loss": "mse",
        "gradient_clipping": True,
        "verbose": 1,
        "base_directory": "../models/",
        "epochs": 10, 
        "model-uuid": "UUID"
    }
    return config_template

#############################
# ML Stuff
#############################

def get_device():
    ''' function to get the device the NN is running on, CPU or GPU
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")
        
    return device


def init_torch(config):
    ''' function for initializing the seeds
        arguments:
            config: the configuration file
    '''
    gen = torch.manual_seed(config['seed'])
    torch.set_default_dtype(torch.float64)
    
    device = get_device()
    if device.type == 'cpu':
        torch.set_num_threads(64)
        # torch.set_num_interop_threads(1)

    logging.info(f' torch is using {device}')
    return device


def lp_loss(p, model):
    ''' function for p-regularization. p = 1 lasso, p = 2 ridge
        argument:
            model: the torch model
    '''
    lp_regularization = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            lp_regularization = lp_regularization + torch.norm(param, p=p)
    return lp_regularization


def train_one_epoch(model, train_data, f_optimizer, f_loss, config):
    ''' function for the training epoch
        arguments:
            model: the pytorch model to nbe trained
            train_data: the training data loader 
            f_optimizer: the optimizer
            f_loss: the loss function
            max_steps: the maximum number of steps in an epoch
            alpha: the weight of the L1 regulatization (default: 0)
            beta: the weight of the L2 regularization (default: 0)
        returns:
            the loss for the epoch
        loss function: 
            loss + beta * (alpha * lp_loss(1, model) + (1 - alpha) * lp_loss(2, model))
    '''
    epoch_loss = 0.

    for i, data in enumerate(train_data):
        
        # steps per epoch
        if i == config["steps_per_epoch"]:
            break
        
        # forward prop
        x, y = data
        f_optimizer.zero_grad()
        y_out = model(x)

        # backprop
        loss = f_loss(y_out, y)
        loss = loss + config["beta"] * (config["alpha"] * lp_loss(1, model) + (1 - config["alpha"]) * lp_loss(2, model))
        if not config["gradient_clipping"]:
            if loss.item() > CLIP:
                return loss.item()
        loss.backward()
        
        # Gradient Norm Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        # Gradient Value Clipping
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        
        f_optimizer.step()

        # Gather data and report
        epoch_loss += loss.item()

    return epoch_loss/(i + 1)


def validate_one_epoch(model, validate_data, f_loss, config):
    ''' function for the validation epoch
        arguments:
            model: the pytorch model to nbe trained
            validate_data: the validation data loader 
            f_loss: the loss function
            max_steps: the maximum number of steps in an epoch
            alpha: the weight of the L1 regulatization (default: 0)
            beta: the weight of the L2 regularization (default: 0)
        returns:
            the loss for the epoch
            the absolute error
            the R2 score
    '''
    epoch_loss = 0.
    
    for i, data in enumerate(validate_data):
        if i == config["steps_per_epoch"]:
            break
        
        # forward prop
        x, y = data
        y_p = model(x)
        
        # loss
        loss = f_loss(y_p, y)
        loss = loss + config["beta"] * (config["alpha"] * lp_loss(1, model) + (1 - config["alpha"]) * lp_loss(2, model))
        epoch_loss += loss.item()
        
        if i == 0:
            y_test = y.cpu().detach().numpy()
            y_pred = y_p.cpu().detach().numpy()
        else: 
            y_test = np.vstack((y_test, y.cpu().detach().numpy()))
            y_pred = np.vstack((y_pred, y_p.cpu().detach().numpy()))
        
    # accuracy
    y_test = np.array(y_test).reshape(-1, config['n_targets'])
    y_pred = np.array(y_pred).reshape(-1, config['n_targets'])
    abs_score = (1 - np.mean(np.abs((y_pred - y_test)/y_test)))*100
    r2_score = metrics.r2_score(y_test, y_pred)*100

    return epoch_loss / (i + 1), abs_score, r2_score


def test_model(model, test_data, config):
    ''' function for testing the model
        arguments:
            model: the pytorch model to nbe trained
            test_data: the test data loader
            config: the configurations file
        returns:
            the absolute error
            the R2 score
    '''
    
    for i, data in enumerate(test_data):
        
        # forward prop
        x, y = data
        y_p = model(x)
        
        # store data
        if i == 0:
            x_test = x.cpu().detach().numpy()
            y_test = y.cpu().detach().numpy()
            y_pred = y_p.cpu().detach().numpy()
        else: 
            x_test = np.vstack((x_test, x.cpu().detach().numpy()))
            y_test = np.vstack((y_test, y.cpu().detach().numpy()))
            y_pred = np.vstack((y_pred, y_p.cpu().detach().numpy()))
        
    
    # accuracy
    y_test = np.array(y_test).reshape(-1, config['n_targets'])
    y_pred = np.array(y_pred).reshape(-1, config['n_targets'])
    abs_score = (1 - np.mean(np.abs((y_pred - y_test)/y_test)))*100
    r2_score = metrics.r2_score(y_test, y_pred)*100
    
    # save test results
    df_pred = pd.DataFrame(x_test, columns=['x'+str(i+1) for i in range(config['input_shape'])])
    
    # unscale the target
    scaled_cname = [s + '_scaled' for s in config['var_y']]
    df_pred = pd.concat([df_pred, pd.DataFrame(y_test, columns=scaled_cname)], axis=1)
    for col in df_pred.columns[4:]:
        df_pred[col[:-7]] = df_pred[col].apply(lambda y: y_unscale(y))
    # y_test_real = pd.DataFrame(np.vectorize(y_unscale)(y_test), columns=config['var_y']) # 
    # df_pred = pd.concat([df_pred, y_test_real], axis=1)
    y_test_real = df_pred.iloc[:,-config['n_targets']:].to_numpy()
    
    # stuff in the scaled predictions
    pred_cname = [s + '_scaled_pred' for s in config['var_y']]
    df_pred = pd.concat([df_pred, pd.DataFrame(y_pred, columns=pred_cname)], axis=1)
    
    # stuff in the unscaled predictions
    pred_cname = [s + '_pred' for s in config['var_y']]
    for col in df_pred.columns[-config['n_targets']:]:
        df_pred[col[:-12]+'_pred'] = df_pred[col].apply(lambda y: y_unscale(y))
    # y_pred_real = pd.DataFrame(np.vectorize(y_unscale)(y_pred), columns=pred_cname) # .apply(lambda y: y_unscale(y))
    # df_pred = pd.concat([df_pred, y_pred_real], axis=1)
    y_pred_real = df_pred.iloc[:,-config['n_targets']:].to_numpy()
    
    # calculate the deltas for the scaled and unscaled targets
    scaled_delta_cname = ['scaled_delta_' + s for s in config['var_y']]
    df_pred = pd.concat([df_pred, pd.DataFrame((y_pred - y_test)/y_test*100, columns=scaled_delta_cname)], axis=1)
    delta_cname = ['delta_' + s for s in config['var_y']]
    df_pred = pd.concat([df_pred, pd.DataFrame((y_pred_real - y_test_real)/y_test_real*100, columns=delta_cname)], axis=1)
    
    #same the whole dataframe
    df_pred.to_csv(config['directory']+'/test-results-'+config['model-uuid']+'-'+f'{abs_score:.6f}-{r2_score:.6f}.csv')
    
    # plot all the scaled and real deltas
    for c in scaled_delta_cname:
        make_error_plot(config, df_pred, col=c)
    for c in delta_cname:
        make_error_plot(config, df_pred, col=c)
        
    config['test_metrics'] = {}
    config['test_metrics']['r2'] = {}
    config['test_metrics']['abs_score'] = {}
    num_vars = len(config['var_y'])
    base_length = config['input_shape'] + num_vars
    for i in range(base_length, base_length + num_vars):
        test = df_pred.iloc[:,i].values
        pred = df_pred.iloc[:,i + 2 * num_vars].values
        config['test_metrics']['r2'][config['var_y'][i - base_length]] = metrics.r2_score(test, pred)*100
        config['test_metrics']['abs_score'][config['var_y'][i - base_length]] = 100 - np.abs(df_pred.iloc[:,i + 4 * num_vars].mean())
    
    config['test_metrics']['r2']['model'] = r2_score
    config['test_metrics']['abs_score']['model'] = abs_score
    
    return abs_score, r2_score


### CHANGES ON JULY 2024 - HARISH 07242024

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Saves the current model state, optimizer state, and scheduler state along with the epoch number to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): The current epoch number.
        path (str): The file path where the checkpoint will be saved.

    Returns:
        None
    """
    # Save the model, optimizer, and scheduler state dictionaries in the given path
    torch.save({
        'epoch': epoch,  # Save the current epoch
        'model_state_dict': model.state_dict(),  # Save the model's state
        'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state
        'scheduler_state_dict': scheduler.state_dict(),  # Save the scheduler's state
    }, path)
    logging.info("Saved state to checkpoint: " + path)


def load_checkpoint(model, optimizer, scheduler, path):
    """
    Loads the model, optimizer, and scheduler states from a checkpoint file, and restores the training process.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to load the state into.
        path (str): The file path from which the checkpoint will be loaded.
x
    Returns:
        int: The epoch to resume training from.
    """
    # Load the saved checkpoint from the specified path
    checkpoint = torch.load(path)

    # Load the state dictionaries into the model, optimizer, and scheduler
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info("Loaded checkpoint state from: " + path)
    
    # Continue training from the next epoch
    logging.info(f"Continuing at epoch: {checkpoint['epoch'] + 1}")

    # Get the current learning rate from the scheduler
    current_lr = scheduler.get_last_lr()
    logging.info(f'Current learning rate: {current_lr}')

    # Return the last epoch to continue training
    return checkpoint['epoch']
#### CHANGES - HARISH END


def runML(df, config):
    """ run the DNN
        arguments:
            df: the dataframe including, training, validation and test
            config: the configuration dictionary for the hyperparameters
    """

    logging.info(' running the regressor')
    train_loader, validation_loader, test_loader = build_data_loaders(df,config)
    
    regressor = nets(config)

    # print the summary
    logging.info('\n' + pms.summary(regressor, torch.zeros((config["input_shape"],)).to(get_device()).double().clone().detach().requires_grad_(True)))
    
    # define the loss function
    if config['loss'] == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif config['loss'] == 'mae':
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError('loss type not defined. Has to be mae or mse')
    
    # define the optimizer
    initial_lr = 0.001
    if 'initial_lr' in config: 
        initial_lr = config['initial_lr']
    optimizer = torch.optim.Adam(regressor.parameters(), lr=initial_lr)
    
    # learning rate decay
    if config['lr_decay_type'] == 'exp':
        gamma = (0.5 ** (10000000 / config["decay_steps"])) ** (1 / 2500)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer,
                        gamma=gamma
                    )
    elif config['lr_decay_type'] == 'lambda_exp':
        # Define decay steps and decay rate
        decay_steps = 50 * config['steps_per_epoch']
        decay_rate = 0.7

        # Lambda function for exponential decay
        lambda_lr = lambda epoch: decay_rate ** (epoch / decay_steps)

        # Learning rate scheduler using LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    elif config['lr_decay_type'] == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
                        optimizer=optimizer,
                        total_iters=config['decay_steps'],
                        power=0.5
                    )
    elif config['lr_decay_type'] == 'const':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer=optimizer,
                        factor=1., 
                        total_iters=config["epochs"]
                    )
    else:
        raise ValueError('lr type not defined. Has to be exp or poly or const')
        
    # the training history
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "abs_score": [], "r2_score": []}
    
    # model storage
    model_path = config['directory'] + f'/dnn-{config["depth"]}-{config["width"]}-{config["activation"]}-adam-{config["lr_decay_type"]}-schedule-{config["loss"]}-{config["monitor"]}.torch'
    
    # early stopping
    early_stopping = EarlyStopping(model_path, patience=config["patience"])
    
    
    # run the regressor
    start = time.time()
    epoch = 0

    ###### CHANGE - HARISH - Load checkpoint 07242024
    checkpoint_base = "../checkpoints"
    checkpoint_dir = os.path.join(checkpoint_base, config['model-uuid'])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    logging.info(" Model checkpoints will be stored in: " + checkpoint_path)

    if "load_checkpoint" in config:
        load_chk_id = config["load_checkpoint"]
        load_chk_dir = os.path.join(checkpoint_base, load_chk_id)
        load_chk_path = os.path.join(load_chk_dir, "checkpoint.pth")
        
        if os.path.exists(load_chk_path):
            epoch = load_checkpoint(regressor, optimizer, scheduler, load_chk_path)
        else:
            logging.error(" The provided checkpoint path does not exist!")
    ###### CHANGE - HARISH - END

    while epoch < config["epochs"]:

        # Make sure gradient tracking is on, and do a pass over the data
        regressor.train(True)
        avg_loss = train_one_epoch(regressor, 
                                   train_loader, 
                                   optimizer, 
                                   loss_fn, 
                                   config)
        
        # for some activations there are spikes in the loss function: skip the epoch when that happens
        if not config["gradient_clipping"]:
            if avg_loss > CLIP and os.path.exists(model_path):
                regressor.train(False)
                early_stopping.reset_counter()
                regressor.load_state_dict(torch.load(model_path))
                continue
            
        # add to history
        history["epoch"].append(epoch + 1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(avg_loss)

        # We don't need gradients on to do reporting
        regressor.train(False)
        avg_vloss, abs_score, r2_score = validate_one_epoch(regressor, 
                                                            validation_loader, 
                                                            loss_fn, 
                                                            config)
        history["val_loss"].append(avg_vloss)
        history["abs_score"].append(abs_score)
        history["r2_score"].append(r2_score)
        
        logging.info(f' Epoch {epoch + 1}: training loss = {avg_loss:.8f}  validation loss = {avg_vloss:.8f}  learning rate = {optimizer.param_groups[0]["lr"]:0.3e}  relative accuracy: {abs_score:.2f}  R2 score: {r2_score:.2f}')
        
        # pass information to config
        config['final_decayed_lr'] = optimizer.param_groups[0]["lr"]
        config['final_validation_r2'] = r2_score
        config['final_validation_loss'] = avg_vloss
        config['last_epoch'] = epoch
        
        # check for early stopping
        if early_stopping.early_stop(regressor, avg_vloss, epoch, config):
            regressor.load_state_dict(torch.load(model_path))
            #### CHECKPOINT CHANGE HARISH 07242024
            save_checkpoint(regressor, optimizer, scheduler, epoch+1, checkpoint_path)
            break

        # if early stopping did not happen revert to the best model
        if epoch == config["epochs"] - 1:
            regressor.load_state_dict(torch.load(model_path))
            
        # decay learning rate
        scheduler.step()
        
        epoch += 1

        #### CHECKPOINT CHANGE HARISH 07242024
        save_checkpoint(regressor, optimizer, scheduler, epoch, checkpoint_path)
        
    config["fit_time"] = timediff(time.time() - start)
    
    return test_loader, regressor, history



def simple_train(df, config):
    """ run the DNN
        arguments:
            df: the dataframe including, training, validation and test
            config: the configuration dictionary for the hyperparameters
    """

    logging.info(' starting the training')
    train_loader, validation_loader, test_loader = build_data_loaders(df,config)

    regressor = nets(config)
    # print the summary
    #logging.info('\n' + pms.summary(regressor, torch.zeros((config["input_shape"],)).to(get_device()).double().clone().detach().requires_grad_(True)))
    
    # define the loss function
    loss_fn_dict = {'mse': torch.nn.MSELoss(), 'mae': torch.nn.L1Loss()}
    loss_fn = loss_fn_dict[config['loss']]

    # define the optimizer
    initial_lr = config['initial_lr'] if 'initial_lr' in config else 0.001
    optimizer = torch.optim.Adam(regressor.parameters(), lr=initial_lr)
    
    # learning rate decay
    if config['lr_decay_type'] == 'exp':
        gamma = (0.5 ** (10000000 / config["decay_steps"])) ** (1 / 2500)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer,
                        gamma=gamma
                    )
    elif config['lr_decay_type'] == 'lambda_exp':
        # Define decay steps and decay rate
        decay_steps = 50 * config['steps_per_epoch']
        decay_rate = 0.7
        # Lambda function for exponential decay
        lambda_lr = lambda epoch: decay_rate ** (epoch / decay_steps)
        # Learning rate scheduler using LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    elif config['lr_decay_type'] == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
                        optimizer=optimizer,
                        total_iters=config['decay_steps'],
                        power=0.5
                    )
    elif config['lr_decay_type'] == 'const':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer=optimizer,
                        factor=1., 
                        total_iters=config["epochs"]
                    )
    else:
        raise ValueError('lr type not defined. Has to be exp or poly or const')
        
    # the training history
    #history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "abs_score": [], "r2_score": []}
    # model storage
    #model_path = config['directory'] + f'/dnn-{config["depth"]}-{config["width"]}-{config["activation"]}-adam-{config["lr_decay_type"]}-schedule-{config["loss"]}-{config["monitor"]}.torch'
    # early stopping
    #early_stopping = EarlyStopping(model_path, patience=config["patience"])
    
    
    # run the regressor
    start = time.time()
    epoch = 0

    while epoch < config["epochs"]:

        # Make sure gradient tracking is on, and do a pass over the data
        regressor.train(True)
        avg_loss = train_one_epoch(regressor, 
                                   train_loader, 
                                   optimizer, 
                                   loss_fn, 
                                   config)

            
        # We don't need gradients on to do reporting
        regressor.train(False)
        avg_vloss, abs_score, r2_score = validate_one_epoch(regressor, 
                                                            validation_loader, 
                                                            loss_fn, 
                                                            config)

        logging.info(f' Epoch {epoch + 1}: training loss = {avg_loss:.8f}  validation loss = {avg_vloss:.8f}  learning rate = {optimizer.param_groups[0]["lr"]:0.3e}  relative accuracy: {abs_score:.2f}  R2 score: {r2_score:.2f}')
        
        # pass information to config
        config['final_decayed_lr'] = optimizer.param_groups[0]["lr"]
        config['final_validation_r2'] = r2_score
        config['final_validation_loss'] = avg_vloss
        config['last_epoch'] = epoch
                    
        # decay learning rate
        scheduler.step()
        
        epoch += 1
    
    return test_loader, regressor

def setup_training_components(config):
    """
    Sets up the model, loss function, optimizer, and learning rate scheduler.

    Args:
        config (dict): Configuration dictionary containing settings for input shape, loss function, 
                       learning rate, decay type, decay steps, and other parameters.

    Returns:
        Tuple containing:
            - regressor (torch.nn.Module): Initialized model.
            - loss_fn (torch.nn.Module): Loss function based on config.
            - optimizer (torch.optim.Optimizer): Optimizer for model training.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    """
    # Initialize the model
    regressor = nets(config)

    # Define the loss function
    loss_fn_dict = {'mse': torch.nn.MSELoss(), 'mae': torch.nn.L1Loss()}
    loss_fn = loss_fn_dict[config.get('loss', 'mse')]  # Default to 'mse' if not specified

    # Define the optimizer
    initial_lr = config.get('initial_lr', 0.001)  # Default initial learning rate to 0.001 if not specified
    optimizer = torch.optim.Adam(regressor.parameters(), lr=initial_lr)

    # Set up learning rate scheduler based on config
    if config.get('lr_decay_type') == 'exp':
        gamma = (0.5 ** (10000000 / config["decay_steps"])) ** (1 / 2500)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif config.get('lr_decay_type') == 'lambda_exp':
        decay_steps = 50 * config['steps_per_epoch']
        decay_rate = 0.7
        lambda_lr = lambda epoch: decay_rate ** (epoch / decay_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    elif config.get('lr_decay_type') == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer, total_iters=config['decay_steps'], power=0.5)
    elif config.get('lr_decay_type') == 'const':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1., total_iters=config["epochs"])
    else:
        raise ValueError('lr_decay_type not defined. Must be one of: "exp", "lambda_exp", "poly", or "const"')

    return regressor, loss_fn, optimizer, scheduler

def objective(trial, config, train_loader, validation_loader, test_loader):
    
    best_loss = float("Inf")

    #config = get_config()


    # Suggest integer values for width, depth, and num_modules
    width = trial.suggest_int("width", 24, 36)
    depth = trial.suggest_int("depth", 1, 16)
    skip_block_layers = trial.suggest_int("skip_block_layers", 1, 15)

    config["width"] = width
    config["depth"] = depth
    config["skip_block_layers"] = skip_block_layers
    
    regressor, loss_fn, optimizer, scheduler = setup_training_components(config)

    trial_number = trial.number
    logging.info(f" Trial #{trial_number}. Params: {trial.params}")

    # print the summary
    # logging.info('\n' + pms.summary(regressor, torch.zeros((config["input_shape"],)).to(get_device()).double().clone().detach().requires_grad_(True)))
    model_parameters = sum(p.numel() for p in regressor.parameters())
    logging.info(f" Total number of parameters: {model_parameters}")


    ########################
    # train the regressor
    start = time.time()
    epoch = 0

    while epoch < config["epochs"]:

        # Make sure gradient tracking is on, and do a pass over the data
        regressor.train(True)
        avg_loss = train_one_epoch(regressor, train_loader, optimizer, loss_fn, config)

            
        # We don't need gradients on to do reporting
        regressor.train(False)
        avg_vloss, abs_score, r2_score = validate_one_epoch(regressor, validation_loader, loss_fn, config)

        if float(avg_vloss) < best_loss:
            best_loss = avg_vloss
        
        trial.report(avg_vloss, epoch)

        # Handle pruning based on the intermediate value.
        #if trial.should_prune():
        #    raise optuna.exceptions.TrialPruned()

        logging.info(f' Epoch {epoch + 1}: training loss = {avg_loss:.8f}  validation loss = {avg_vloss:.8f}  learning rate = {optimizer.param_groups[0]["lr"]:0.3e}  relative accuracy: {abs_score:.2f}  R2 score: {r2_score:.2f}')
        # decay learning rate
        scheduler.step()
        
        epoch += 1

    ########################

    return best_loss
     
def perform_trials(df, config):
    # prints the best trial value after the end of all trials
    def print_best_callback(study, trial):
        logging.info(f" Best value: {study.best_value}, Best params: {study.best_trial.params}")

    #config = get_config()
    # Build data loaders
    train_loader, validation_loader, test_loader = build_data_loaders(df, config)

    # we create an optimization study object which will search for the best parameters and store them
    study = optuna.create_study(direction="minimize")
    # this will optimize the objective function by performing 35 trials. at the end of each trial it calls the callback function
    study.optimize(
        lambda trial: objective(trial, config, train_loader, validation_loader, test_loader), 
        n_trials=1000, timeout=28000, callbacks=[print_best_callback])

    # pruned trials are those which do not seem to get optimal results, so optuna will terminate the training process for that trial
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logging.info(" Study statistics: ")
    logging.info("   Number of finished trials: ", len(study.trials))
    logging.info("   Number of pruned trials: ", len(pruned_trials))
    logging.info("   Number of complete trials: ", len(complete_trials))

    logging.info(" Best trial:")
    trial = study.best_trial

    logging.info("   Value: ", trial.value)

    logging.info("   Params: ")
    for key, value in trial.params.items():
        logging.info("     {}: {}".format(key, value))

    # Assuming 'study' is your Optuna study object
    study_save_path = '../models/optuna_study.pkl'
    # To SAVE the study
    with open(study_save_path, 'wb') as f:
        pickle.dump(study, f)


     
   

#############################
# utils
#############################

def timediff(x):
    """ a function to convert seconds to hh:mm:ss
        argument:
            x: time in seconds
        returns:
            time in hh:mm:ss
    """
    return "{}:{}:{}".format(int(x/3600), str(int(x/60%60)).zfill(2), str(round(x - int(x/3600)*3600 - int(x/60%60)*60)).zfill(2))


def plot_loss(history, dir_name):
    """ plotting routine
        argument:
            history: the tf history object
    """
    plt.plot(history['train_loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('y')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/training-evaluation.pdf', dpi=300)
    plt.close()
    
    
def labeler(config) -> str:
    return '{}-{} ({:,})'.format(config['depth'], config['width'], config['trainable_parameters'])


def make_error_plot(config, df, col):
    lim = df[col].std()*5
    lim = lim if np.isfinite(lim) else 1e6
    plt.figure(figsize=(8,6))
    plt.hist(df[abs(df[col]) < lim][col], bins=200, histtype='step', linewidth=3, density=True, label=labeler(config))
    plt.xlabel(r'$\delta$ (\%)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim((-lim, lim))
    plt.grid(linestyle='dashed', alpha=0.4, color='#808080')
    plt.legend(fontsize=16)
    plt.title(r'$\delta$ distribution [mean: {:.4f}%, std: {:.4f}%]'.format(df[col].mean(), df[col].std()), fontsize=20)
    plt.tight_layout()
    plt.savefig(config['directory']+'/'+col+'-distribution.pdf', dpi=300)
    plt.close()


def post_process(regressor, test_loader, history, config):
    """ post process the regressor to check for accuracy and save everything
        argumants:
            regressor: the tensorflow regressor object
            history: the history object for the training
            config: the configuration for the training
    """
    logging.info(' running post-process')
    
    # check accuracy
    logging.info(' running the DNN predictions and accuracy computation')
    
    abs_score, r2_score = test_model(regressor, test_loader, config)
    logging.info(' relative accuracy: {:.2f}%  |---|  R2 score: {:.2f}%'.format(abs_score, r2_score))
        
    #plot the training history
    logging.info(' printing training evaluation plots')
    plot_loss(history, config['directory'])
    
    # end time
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
                 
    # save config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # save history
    with open(config['directory']+'/history-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    # remove preliminary config file
    os.remove(config['directory']+'/config-'+config['model-uuid']+'-prelim.json')
        
    # move directory
    mc = 'MC' if config['use_MC_sample'] else ''
    shutil.move(config['directory'], config['directory']+'-'+mc+'-'+str(config['depth'])+'-'+str(config['width'])+'-'+config['activation']+'-'+str(config['batch_size'])+'-adam-'+config['lr_decay_type']+'-schedule-'+config['loss']+'-'+config['monitor']+f'-{abs_score:.6f}-{r2_score:.6f}') 
    
    
#############################
# main
#############################
    
def main():
    
    config = get_config()
        
    # init torch
    device = init_torch(config)
    
    # start time
    config["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    
    # set device
    config["device"] = device.type + ':' + device.index if device.index else device.type
    
    #  create directory structure
    if config['model-uuid'] == "UUID":
        m_uuid = str(uuid.uuid4())[:8]
        config['model-uuid'] = m_uuid
    else:
        m_uuid = config['model-uuid']
        
        
    # load data
    df, spark = load_data(config)

    perform_trials(df, config)
    
    logging.info(' stopping Spark session')
    spark.stop()


if __name__ == "__main__":
    # execute only if run as a script
    main()
