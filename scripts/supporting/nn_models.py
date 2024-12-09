import numpy as np
import torch
import logging
import uuid
import os
import shutil
import json
import argparse
import pytorch_model_summary as pms

def get_device():
    ''' function to get the device the NN is running on, CPU or GPU
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")
        
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
            # reset the counter everytime a new best validation loss is encountered
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                torch.save(model.state_dict(), self.m_path)
                config['n_training_epochs'] = epoch
                self.counter = 0
            # increment everytime the loss does not go down and signal early stopping when the patience is crossed
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
        # Initializing weights with Glorot (Xavier) normal and biases to zero
        torch.nn.init.xavier_normal_(self.input.weight)
        torch.nn.init.zeros_(self.input.bias)

        # create a sequence of linear layers
        self.fc_module = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) for i in range(n_layers)])
        # Apply Xavier (Glorot) normal initialization to layers of fc_module
        for layer in self.fc_module:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        # if stream is true, the output layer is reshaped to input_shape, or else 
        # the hidden layer width is maintained constant
        if self.stream:
            self.linear = torch.nn.Linear(self.width, self.input_shape)
        else:
            self.linear = torch.nn.Linear(self.width, self.width)
        # Initializing weights with Glorot (Xavier) normal and biases to zero
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
        # if input_shape is not equal to width, then add a reshape layer to reshape input_shape to width
        if self.input_shape != self.width:
            self.reshape = torch.nn.Linear(self.input_shape, self.width, bias=False)
            torch.nn.init.xavier_normal_(self.reshape.weight)
        
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
        
        
def getActivation(config):
    ''' function for defining the activation function
        argument:
            config: the configurations file
        returns
            the specified activations function
    '''
    if config["activation"] == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif config["activation"] == 'relu':
        return torch.nn.ReLU()
    if config["activation"] == 'softplus':
        return torch.nn.Softplus()
    if config["activation"] == 'swish':
        return torch.nn.SiLU()
    if config["activation"] == 'sigmoid':
        return torch.nn.Sigmoid()
    if config["activation"] == 'tanh':
        return torch.nn.Tanh()
    if config["activation"] == 'prelu':
        return torch.nn.PReLU()
        
    
class skip_dnn(torch.nn.Module):
    ''' class for the DNN with skip connections: see https://arxiv.org/abs/2302.00753
    '''
    def __init__(self, sk_block, config, stream = False):
        super(skip_dnn, self).__init__()
        self.width = config["width"]
        self.n_blocks = config["depth"] - 1
        self.input_shape = config["input_shape"]
        self.output_shape = config['n_targets']
        self.n_layers = config['skip_block_layers']
        self.stream = stream
        self.act = getActivation(config)
        
        # input data is passed to a skip block
        self.input = skip_block(self.input_shape, 
                                self.width, 
                                self.act, 
                                stream = self.stream, 
                                n_layers = self.n_layers)
        self.core = self.make_layers(skip_block)
        
        # based on the value of stream, last layer input can be equal to 'input_shape' or 'width'. 
        if self.stream: 
            self.output = torch.nn.Linear(self.input_shape, self.output_shape)
        else: 
            self.output = torch.nn.Linear(self.width, self.output_shape)
        
        # Initializing weights with Glorot (Xavier) normal and biases to zero
        torch.nn.init.xavier_normal_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
            
    def make_layers(self, skip_block):
        '''
        create a sequence of skip blocks for the core of the skip dnn
        '''
        layers = []
        for bl in range(self.n_blocks):
            if self.stream: 
                layers.append(skip_block(self.input_shape, 
                                         self.width, 
                                         self.act, 
                                         stream=self.stream, 
                                         n_layers = self.n_layers))
            else:
                layers.append(skip_block(self.width, 
                                         self.width, 
                                         self.act, 
                                         n_layers = self.n_layers))
            
        return torch.nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.input(x)
        x = self.core(x)
        return self.output(x)
    
    
class dnn(torch.nn.Module):
    ''' class for a feed-forward DNN
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

class skip_light_module(torch.nn.Module):
    """ 
        This is a skip dnn component of the skip_light neural network 
        argument:
            config: the configurations file
    """
    def __init__(self, config):
        super(skip_light_module, self).__init__()
        self.width = config["width"]
        self.skip_layer_depth = config["skip_block_layers"]
        self.act = getActivation(config)
        self.fc_module = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) 
                                              for i in range(self.skip_layer_depth)])
        # Apply Xavier (Glorot) normal initialization to layers of fc_module
        for layer in self.fc_module:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        y = x
        # vector x shape and width are the same
        for layer in self.fc_module:
            y = self.act(layer(y))
        y = y+x
        return y
    
class skip_light(torch.nn.Module):
    """
        implementation of the skip network as a lighter version of the skip_dnn, based on Fady's implementation
        argument:
            config: the configurations file
    """
    def __init__(self, config):
        super(skip_light, self).__init__()
        self.input_shape = config["input_shape"]
        self.width = config["width"]
        self.n_modules = config["depth"]
        self.skip_depth = config["skip_block_layers"]
        self.output_shape = config['n_targets']
        
        # input layer
        self.input = torch.nn.Linear(self.input_shape, self.width)
        torch.nn.init.xavier_normal_(self.input.weight)  # Equivalent to 'glorot_normal'
        torch.nn.init.zeros_(self.input.bias)  # Equivalent to 'zeros'
        self.act = getActivation(config)
        #skip blocks
        self.skip_core = torch.nn.Sequential(*[
            skip_light_module(config) 
            for _ in range(self.n_modules)
        ])
        #output layer
        self.output = torch.nn.Linear(self.width, self.output_shape)
        torch.nn.init.xavier_normal_(self.output.weight)  # Equivalent to 'glorot_normal'
        torch.nn.init.zeros_(self.output.bias)  # Equivalent to 'zeros'
        
    def forward(self, x):
        x = self.act(self.input(x))
        x = self.skip_core(x)
        x = self.output(x)
        return x
    
    
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
        regressor = skip_dnn(skip_block, config).double().to(get_device())
    elif config["model_type"] == 'skip-stream':
        regressor = skip_dnn(skip_block, config, stream = True).double().to(get_device())
    elif config["model_type"] == 'skip-light':
        regressor = skip_light(config).double().to(get_device())
    else:
        logging.error(' '+config["model_type"]+' not implemented. model_type can be either dnn, skip or squeeze')
        
        
    # save parameter counts
    summary = pms.summary(regressor, torch.zeros((config["input_shape"],)).to(get_device()).double().clone().detach().requires_grad_(True)).rstrip().split('\n')
    config["trainable_parameters"] = int(summary[-3].replace(',', '')[18:])
    config["non_trainable_parameters"] = int(summary[-2].replace(',', '')[22:])
    config["total_parameters"] = int(summary[-4].replace(',', '')[14:])
    
    # save config
    # with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
    #     json.dump(config, f, indent=4)
        
    return regressor

