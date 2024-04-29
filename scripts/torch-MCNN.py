#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark.sql.types import StructType, StructField, DoubleType
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
from collections import OrderedDict

from matplotlib import rc

rc('text', usetex=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CLIP = 1e12


def load_data(config):
    """ Load the data using pyspark
        arguments:
            config: the configuration file
        returns:
            df['train']: the training dataframe
            df['validate']: the validation data set
            df['test']: the testing data set
            spark: the spark session
    """
    
    # Spark session and configuration
    logging.info(' creating Spark session')
    spark = (SparkSession.builder.master("local[48]")
             .config('spark.executor.instances', 16)
             .config('spark.executor.cores', 16)
             .config('spark.executor.memory', '10g')
             .config('spark.driver.memory', '15g')
             .config('spark.memory.offHeap.enabled', True)
             .config('spark.memory.offHeap.size', '20g')
             .config('spark.dirver.maxResultSize', '20g')
             .config('spark.debug.maxToStringFields', 500)
             .appName("amp.hell").getOrCreate())

    # Enable Arrow-based columnar data 
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set(
        "spark.sql.execution.arrow.pyspark.fallback.enabled", "true"
    )
    logging.info(' Spark initialized')
    
    # read the data into a spark frame
    start = time.time()
    path = config['data_dir']
    if path[-1] != '/':
        path = path + '/'
    input_var = ['x'+str(i+1) for i in range(config['input_shape'])]
    target_var = ['y'+str(i+1) for i in range(config['amplitude_shape'])]
    header = input_var + target_var
    schema = StructType([StructField(header[i], DoubleType(), True) for i in range(config['input_shape']+config['amplitude_shape'])])
    df = {}
    df['train'] = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+'train/*.csv.*', header='true')
    
    if config['var_y'] == 'all': config['var_y'] = df['train'].columns[config['input_shape']:]
    
    if config['use_MC_sample']:
        mc_suffix = '_mc'
    else:
        mc_suffix = ''
    df['validate'] = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+'validate'+mc_suffix+'/*.csv.*', header='true')
    df['test'] = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+'test'+mc_suffix+'/*.csv.*', header='true')

    logging.info(' data loaded into Spark session in {:.3f} seconds'.format(time.time() - start))
    
    # transfer the data to a pandas dataframe
    start = time.time()
    train_sample = min(config['train-sample-size'], df['train'].count())
    validate_sample = min(config['validate-sample-size'], df['validate'].count())
    test_sample = min(config['test-sample-size'], df['test'].count())
        
    df['train'] = df['train'].select(*input_var, *config['var_y']).limit(train_sample).toPandas() 
    df['validate'] = df['validate'].select(*input_var, *config['var_y']).limit(validate_sample).toPandas()
    df['test'] = df['test'].select(*input_var, *config['var_y']).limit(test_sample).toPandas()
    
    logging.info(' training data shape: {} x {}'.format(df['train'].shape[0], df['train'].shape[1]))
    logging.info(' validation data shape: {} x {}'.format(df['validate'].shape[0], df['validate'].shape[1]))
    logging.info(' testing data shape: {} x {}'.format(df['test'].shape[0], df['test'].shape[1]))
    
    config['train-samples-used'] = df['train'].shape[1]
    config['validate-samples-used'] = df['validate'].shape[1]
    config['test-samples-used'] = df['test'].shape[1]

    logging.info(' data loaded into pandas dataframe in {:.3f} seconds'.format(time.time() - start))
    
    logger = spark._jvm.org.apache.log4j
    logging.getLogger("py4j.clientserver").setLevel(logging.WARN)
    
    return df, spark


def normalize(df, config, var_y):
    """ a function to normaliza the target distribution
        arguments:
            df: dataframe containing the target variable
            config: the config file with the run configuration
            var_y: the variable to be normalized
    """
    config['scaling'][var_y] = {}
    config['scaling'][var_y]["mu"] = df[var_y].mean()
    config['scaling'][var_y]["sigma"] = df[var_y].std()
    
    return (df[var_y] - config['scaling'][var_y]["mu"])/config['scaling'][var_y]["sigma"]


def x_scale(x, p=7.5):
    ''' function for scaling x1
        argument:
            x: the input variable
            p: the scaling factor (default: 7.5)
        returns:
            the scaled variable
    '''
    return 1/p * np.log(1 + x * (np.exp(p) - 1))
                        
    
def y_scale(y):
    ''' function for scaling x1
        argument:
            y: the input variable
        returns:
            the scaled variable
    '''
    return np.log(1 + y) if y >= 0 else -np.log(1 - y)


def y_unscale(y):
    ''' function for scaling x1
        argument:
            y: the input variable
        returns:
            the scaled variable
    '''
    return np.exp(y) - 1 if y >= 0 else 1 - np.exp(-y)


class df_to_tensor(torch.utils.data.Dataset):
    ''' class to convert dataframe to torch tensor
    '''
 
    def __init__(self, df, config):
        self.df = df.copy(deep = True)
        
        # extract x and scale
        x = df.iloc[:, :config['input_shape']]
        df['x1'] = df['x1'].apply(lambda x: x_scale(x))
        
        # extract y
        y = df[config['var_y']]
        
        # number of targets
        config['n_targets'] = y.shape[1]
            
        # scale y
        config['scaling'] = {}
        for column in y.columns:
            y[column] = y[column].apply(lambda y: y_scale(y))
            if config['normal_scaled']:
                y[column] = normalize(y, config, column)

        # put them in tensors. Note: the reshaping of y.
        self.x = torch.tensor(x.values, dtype=torch.float64).to(get_device())
        self.y = torch.tensor(y.values, dtype=torch.float64).reshape(-1, config['n_targets']).to(get_device())
 
    def __len__(self):
        return len(self.x)
   
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    
def build_data_loaders(df, config):
    ''' function to build the data loaders
        arguments:
            df: dictionary of pandas dataframe for test, validation, and train
            config: configuration file
        returns:
            train_loader: the training set loader
            validation_loader: the validation set loader
            test_loader: the test set loader
    '''
    
    # slice and dice the data into train, validation and test sets
    df_test_data =  df['test']
    df_validation_data = df['validate']
    df_train_data = df['train']

    test_data = df_to_tensor(df_test_data, config)
    validation_data = df_to_tensor(df_validation_data, config)
    train_data = df_to_tensor(df_train_data, config)

    # load the data into torch tensor batches
    batch = config["batch_size"]
    numworkers = 2 if get_device().type == 'cpu' else 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=numworkers, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch, shuffle=True, num_workers=numworkers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False, num_workers=numworkers, drop_last=True)
    
    return train_loader, validation_loader, test_loader


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


class EarlyStopping:
    ''' class for early stopping with patience
    '''
    def __init__(self, m_path, patience=1, min_delta=0):
        '''
            arguments:
                m_path: the model path where the checkpoints go
                patience: the number of epochs the patience lasts
                min_delta: the minimum change that is monitored
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.m_path = m_path
        
    def reset_counter(self):
        ''' function to reset the counter
        '''
        self.counter = 0

    def early_stop(self, model, validation_loss, epoch, config):
        ''' function to check for the early stopping threshold
            arguments:
                model: the torch model
                validation_loss: the validation loss
        '''
        if epoch > config['early_stopping_start_epoch']: 
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                torch.save(model.state_dict(), self.m_path)
                config['n_training_epochs'] = epoch
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        else:
            return False
    
    
class skip_block(torch.nn.Module):
    """ the basic building block of a dnn with skip connections
        arguments:
            x: the input
            width: the width of the hidden layers
            activation: the activation function: 'relu', 'elu', 'swish' (silu), 'leaky_relu', 'softplus'
            squeeze: a boolean specifying wheher the skip units are squeezed
        returns:
            res: the skip net block
    """
    def __init__(self, input_shape, width, activation, stream = False, n_layers=1):
        ''' init function
            arguments:
                input_shape: the input shape of the block
                width: the width of the block
                activation: the activation function
                stream: build a residual stream or not (default: false)
                n_layers: the number of layers in the block (default: 1)
        '''
        super(skip_block, self).__init__()
        self.input_shape = input_shape
        self.width = width
        self.stream = stream
        
        self.input = torch.nn.Linear(self.input_shape, self.width)
        self.fc_module = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) for i in range(n_layers)])
        
        if self.stream:
            self.linear = torch.nn.Linear(self.width, self.input_shape)
        else:    
            self.linear = torch.nn.Linear(self.width, self.width)
            
        if self.input_shape != self.width:
            self.reshape = torch.nn.Linear(self.input_shape, self.width, bias=False)
        
        self.act = activation
        
    def forward(self, x):
        ''' forward propagation
        '''
        y = self.act(self.input(x))
        for l in self.fc_module:
            y = self.act(l(y))
        if self.stream:
            y = self.act(self.linear(y))
            y += x
            return y
        else:
            y = self.linear(y)
            if self.input_shape != self.width:
                residual = self.reshape(x)
            else:
                residual = x
            y += residual

            return self.act(y)
        
        
def getActivation(act):
    ''' function for defining the activation function
        argument:
            config: the configurations file
        returns
            the specified activations function
    '''
    if act == 'leaky_relu':
        return torch.nn.LeakyReLU()
    if act == 'relu':
        return torch.nn.ReLU()
    if act == 'softplus':
        return torch.nn.Softplus()
    if act == 'swish':
        return torch.nn.SiLU()
    if act == 'sigmoid':
        return torch.nn.Sigmoid()
        
    
class skip_mcdnn(torch.nn.Module):
    ''' class for the DNN with skip connections: see https://arxiv.org/abs/2302.00753
    '''
    def __init__(self, sk_block, config, stream = False):
        super(skip_mcdnn, self).__init__()
        self.width = config["width"]
        self.n_blocks = config["depth"] - 1
        self.input_shape = config["input_shape"]
        self.output_shape = config['n_col_targets']
        self.n_layers = config['skip_block_layers']
        self.stream = stream
        self.act = getActivation(config['activation'])
        self.n_columns = config['n_columns']
        
        self.mlp_width = config["mlp_width"]
        self.mlp_blocks = config['mlp_blocks']
        self.mlp_input_shape = self.output_shape * self.n_columns
        self.mlp_output_shape = config['n_targets']
        self.mlp_n_layers = config['mlp_skip_block_layers']
        self.mlp_act = getActivation(config['mlp_activation'])
        
        self.input = OrderedDict()
        self.core = OrderedDict()
        self.output = OrderedDict()
        
        for col in range(self.n_columns):
            self.input['input_col'+str(col)] = skip_block(self.input_shape, 
                                                          self.width, 
                                                          self.act, 
                                                          stream = self.stream, 
                                                          n_layers = self.n_layers)
            
            self.core['core_col'+str(col)] = self.make_layers(skip_block)
            if self.stream: 
                self.output['output_col'+str(col)] = torch.nn.Linear(self.input_shape, self.output_shape)
            else: 
                self.output['output_col'+str(col)] = torch.nn.Linear(self.width, self.output_shape)
                
        self.mlp = self.make_layers(skip_block, loc='mlp')
        self.mlp_output = torch.nn.Linear(self.mlp_width, self.mlp_output_shape)
        
            
    def make_layers(self, skip_block, loc='core'):
        layers = OrderedDict()
        l_width = self.width if loc == 'core' else self.mlp_width
        num_blocks = self.n_blocks if loc == 'core' else self.mlp_blocks
        i_shape = self.input_shape if loc == 'core' else self.mlp_input_shape
        o_shape = self.output_shape if loc == 'core' else self.mlp_output_shape
        num_layers = self.n_layers if loc == 'core' else self.mlp_n_layers
        actt = self.act if loc == 'core' else self.mlp_act
        
        for bl in range(num_blocks):
            if self.stream: 
                layers['stream_block_'+str(bl)] = skip_block(i_shape, 
                                                             l_width, 
                                                             actt, 
                                                             stream = self.stream, 
                                                             n_layers = num_layers)
            elif loc == 'core':
                layers['core_block_'+str(bl)] = skip_block(l_width, 
                                                           l_width, 
                                                           actt, 
                                                           n_layers = num_layers)
            else:
                layers['input_block_'+str(bl)] = skip_block(i_shape, 
                                                            l_width, 
                                                            actt, 
                                                            n_layers = num_layers)
            
        return torch.nn.Sequential(layers)
    
        
    def forward(self, x):
        x_f = tuple()
        for col in range(self.n_columns):
            x_tmp = self.input['input_col'+str(col)](x)
            x_tmp = self.core['core_col'+str(col)](x_tmp)
            x_tmp = self.output['output_col'+str(col)](x_tmp)
            x_f += (x_tmp,)
        # print("x_f", x_f)
            
        x_out = torch.cat(x_f, dim=-1)
        # print("x_out", x_out)
        y = self.mlp(x_out)
        return self.mlp_output(y)
    
    
class dnn(torch.nn.Module):
    ''' class for a feed-forward DNN
    TODO: modify for columns
    '''
    def __init__(self, config):
        super(dnn, self).__init__()
        self.width = config["width"]
        
        self.input = torch.nn.Linear(config["input_shape"], self.width)
        self.fc_module = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) for i in range(config["depth"] - 1)])
        self.output = torch.nn.Linear(self.width, config['n_targets'])
        self.act = getActivation(config)
        
    def forward(self, x):
        x = self.act(self.input(x))
        for l in self.fc_module:
            x = self.act(l(x))
        return self.output(x)
    
    
def nets(config):
    """ the pytorch model builder
        arguments:
            config: the configuration file
        returns:
            regressor: the pytorch model
    """
    
    # define the torch model
    if config["model_type"] == 'dnn':
        regressor = dnn(config).double().to(get_device())
    elif config["model_type"] == 'skip':
        regressor = skip_mcdnn(skip_block, config).double().to(get_device())
    elif config["model_type"] == 'skip-stream':
        regressor = skip_mcdnn(skip_block, config, stream = True).double().to(get_device())
    else:
        logging.error(' '+config["model_type"]+' not implemented. model_type can be either dnn, skip or squeeze')
        
        
    # save parameter counts
    summary = pms.summary(regressor, torch.zeros((config["input_shape"],)).to(get_device()).double().clone().detach().requires_grad_(True)).rstrip().split('\n')
    # config["trainable_parameters"] = int(summary[-3].replace(',', '')[18:])
    config["trainable_parameters"] = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    config["non_trainable_parameters"] = sum(p.numel() for p in regressor.parameters() if not p.requires_grad)
    config["total_parameters"] = sum(p.numel() for p in regressor.parameters())
    
    logging.info(f' number of trainable parameters {config["trainable_parameters"]}')
    logging.info(f' number of non-trainable parameters {config["non_trainable_parameters"]}')
    logging.info(f' total number of parameters {config["total_parameters"]}')
    
    # save config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    return regressor
    


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
        
        if config["gradient_clipping"]:
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
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)
    
    # learning rate decay
    if config['lr_decay_type'] == 'exp':
        gamma = (0.5 ** (10000000 / config["decay_steps"])) ** (1 / 2500)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer,
                        gamma=gamma
                    )
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
            break

        # if early stopping did not happen revert to the best model
        if epoch == config["epochs"] - 1:
            regressor.load_state_dict(torch.load(model_path))
            
        # decay learning rate
        scheduler.step()
        
        epoch += 1
        
    config["fit_time"] = timediff(time.time() - start)
    
    return test_loader, regressor, history


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
    
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

    parser = argparse.ArgumentParser(description="A torch implementation of high-precision regressors",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="configuration file for the run")
    args = vars(parser.parse_args())

    # set up the config
    with open(args['config'], 'r') as f:
        config = json.load(f)
        
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
        
    if config['base_directory'] != '':
        base_directory = config['base_directory'] +'/' if config['base_directory'][-1] != '/' else config['base_directory']
        dir_name = base_directory+config['model_type']+'-torch-'+str(config['input_shape'])+'D'+'-'+'_'.join(config['var_y'])+'-'+m_uuid
    else:
        dir_name = config['model_type']+'-torch-'+str(config['input_shape'])+'D'+'-'+'_'.join(config['var_y'])+'-'+m_uuid
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    config['directory'] = dir_name
    
    # save the config
    with open(config['directory']+'/config-'+config['model-uuid']+'-prelim.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # load data
    df, spark = load_data(config)

    # train the regressor
    test_loader, regressor, history = runML(df, config)

    # post process the results and save everything
    post_process(regressor, test_loader, history, config)
    
    logging.info(' stopping Spark session')
    spark.stop()
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()