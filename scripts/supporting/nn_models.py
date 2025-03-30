import numpy as np
import torch
import logging
import uuid
import os
import shutil
import json
import argparse
import pytorch_model_summary as pms
import torch.nn as nn
import torch.nn.functional as F

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
    
    
class skip_block(nn.Module):
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
        
        self.input = nn.Linear(self.input_shape, self.width)
        # Initializing weights with Glorot (Xavier) normal and biases to zero
        nn.init.xavier_normal_(self.input.weight)
        nn.init.zeros_(self.input.bias)

        # create a sequence of linear layers
        self.fc_module = nn.ModuleList([nn.Linear(self.width, self.width) for i in range(n_layers)])
        # Apply Xavier (Glorot) normal initialization to layers of fc_module
        for layer in self.fc_module:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # if stream is true, the output layer is reshaped to input_shape, or else 
        # the hidden layer width is maintained constant
        if self.stream:
            self.linear = nn.Linear(self.width, self.input_shape)
        else:
            self.linear = nn.Linear(self.width, self.width)
        # Initializing weights with Glorot (Xavier) normal and biases to zero
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # if input_shape is not equal to width, then add a reshape layer to reshape input_shape to width
        if self.input_shape != self.width:
            self.reshape = nn.Linear(self.input_shape, self.width, bias=False)
            nn.init.xavier_normal_(self.reshape.weight)
        
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
        return nn.LeakyReLU()
    elif config["activation"] == 'relu':
        return nn.ReLU()
    if config["activation"] == 'softplus':
        return nn.Softplus()
    if config["activation"] == 'swish':
        return nn.SiLU()
    if config["activation"] == 'sigmoid':
        return nn.Sigmoid()
    if config["activation"] == 'tanh':
        return nn.Tanh()
    if config["activation"] == 'prelu':
        return nn.PReLU()
    if config["activation"] == 'elu':
        return torch.nn.ELU()
        
        
    
class skip_dnn(nn.Module):
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
            self.output = nn.Linear(self.input_shape, self.output_shape)
        else: 
            self.output = nn.Linear(self.width, self.output_shape)
        
        # Initializing weights with Glorot (Xavier) normal and biases to zero
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
            
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
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.input(x)
        x = self.core(x)
        return self.output(x)
    
    
class dnn(nn.Module):
    ''' class for a feed-forward DNN
    '''
    def __init__(self, config):
        super(dnn, self).__init__()
        self.width = config["width"]
        
        self.input = nn.Linear(config["input_shape"], self.width)
        self.fc_module = nn.ModuleList([nn.Linear(self.width, self.width) for i in range(config["depth"] - 1)])
        self.output = nn.Linear(self.width, config['n_targets'])
        self.act = getActivation(config)
        
    def forward(self, x):
        x = self.act(self.input(x))
        for l in self.fc_module:
            x = self.act(l(x))
        return self.output(x)

# class skip_light_module(nn.Module):
#     """ 
#         This is a skip dnn component of the skip_light neural network 
#         argument:
#             config: the configurations file
#     """
#     def __init__(self, config):
#         super(skip_light_module, self).__init__()
#         self.width = config["width"]
#         self.skip_layer_depth = config["skip_block_layers"]
#         self.act = getActivation(config)
#         self.fc_module = nn.ModuleList([nn.Linear(self.width, self.width) 
#                                               for i in range(self.skip_layer_depth)])
#         # Apply Xavier (Glorot) normal initialization to layers of fc_module
#         for layer in self.fc_module:
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
        
#     def forward(self, x):
#         y = x
#         # vector x shape and width are the same
#         for layer in self.fc_module:
#             y = self.act(layer(y))
#         y = y+x
#         return y
    
# class skip_light(nn.Module):
#     """
#         implementation of the skip network as a lighter version of the skip_dnn, based on Fady's implementation
#         argument:
#             config: the configurations file
#     """
#     def __init__(self, config):
#         super(skip_light, self).__init__()
#         self.input_shape = config["input_shape"]
#         self.width = config["width"]
#         self.n_modules = config["depth"]
#         self.skip_depth = config["skip_block_layers"]
#         self.output_shape = config['n_targets']
        
#         # input layer
#         self.input = nn.Linear(self.input_shape, self.width)
#         nn.init.xavier_normal_(self.input.weight)  # Equivalent to 'glorot_normal'
#         nn.init.zeros_(self.input.bias)  # Equivalent to 'zeros'
#         self.act = getActivation(config)
#         #skip blocks
#         self.skip_core = nn.Sequential(*[
#             skip_light_module(config) 
#             for _ in range(self.n_modules)
#         ])
#         #output layer
#         self.output = nn.Linear(self.width, self.output_shape)
#         nn.init.xavier_normal_(self.output.weight)  # Equivalent to 'glorot_normal'
#         nn.init.zeros_(self.output.bias)  # Equivalent to 'zeros'
        
#     def forward(self, x):
#         x = self.act(self.input(x))
#         x = self.skip_core(x)
#         x = self.output(x)
#         return x
    

class skip_light_module(nn.Module):
    def __init__(self, config):
        super(skip_light_module, self).__init__()
        self.width = config["width"]
        self.skip_layer_depth = config["skip_block_layers"]
        self.act = getActivation(config)
        
        self.fc_module = nn.ModuleList([
            nn.Linear(self.width, self.width) 
            for _ in range(self.skip_layer_depth)
        ])
        
        self.dropout_rate = config.get("dropout_rate", 0.0)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        
        # Allow a "none" option to disable batch norm.
        self.bn_mode = config.get("bn_mode", "per_layer")
        if self.bn_mode == "per_layer":
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(self.width)
                for _ in range(self.skip_layer_depth)
            ])
        elif self.bn_mode == "per_block":
            self.bn_block = nn.BatchNorm1d(self.width)
        elif self.bn_mode == None:
            # No batch norm is used.
            pass
        else:
            raise ValueError(f"Invalid bn_mode: {self.bn_mode}. Use 'per_layer', 'per_block', or 'None'.")
    
    def forward(self, x):
        y = x
        if self.bn_mode == "per_layer":
            for layer, bn in zip(self.fc_module, self.bn_layers):
                y = self.act(bn(layer(y)))
            y = y + x
        elif self.bn_mode == "per_block":
            for layer in self.fc_module:
                y = self.act(layer(y))
            y = self.bn_block(y)
            y = y + x
        elif self.bn_mode == "none":
            for layer in self.fc_module:
                y = self.act(layer(y))
            y = y + x  # Residual connection is still applied.
        if self.dropout is not None:
            y = self.dropout(y)
        return y

class skip_light(nn.Module):
    """
    Implementation of the skip network (a lighter version of skip_dnn).
    
    Config keys used:
      - "input_shape": input feature dimension.
      - "width": width of hidden layers.
      - "depth": number of skip modules.
      - "skip_block_layers": number of layers per skip module.
      - "n_targets": number of output targets.
      - "activation": activation type (used by getActivation).
      - (Other keys like dropout_rate and bn_mode are passed down to skip_light_module.)
    """
    def __init__(self, config):
        super(skip_light, self).__init__()
        self.input_shape = config["input_shape"]
        self.width = config["width"]
        self.n_modules = config["depth"]
        self.output_shape = config["n_targets"]
        
        # Input layer
        self.input = nn.Linear(self.input_shape, self.width)
        self.act = getActivation(config)
        
        # Skip blocks (each with its own dropout & batch normalization behavior)
        self.skip_core = nn.Sequential(*[
            skip_light_module(config)
            for _ in range(self.n_modules)
        ])
        
        # Output layer
        self.output = nn.Linear(self.width, self.output_shape)
        
    def forward(self, x):
        x = self.act(self.input(x))
        x = self.skip_core(x)
        x = self.output(x)
        return x


class DenseModule(nn.Module):
    """
    A dense module (mini-MLP) with a fixed number of linear layers.
    
    Each layer produces an output of size config["width"]. The module output
    is concatenated with the module input in the overall network.
    
    Args:
        config (dict): Configuration dictionary.
        in_features (int): Number of input features to this module.
    """
    def __init__(self, config, in_features):
        super(DenseModule, self).__init__()
        self.num_layers = config["skip_block_layers"]
        self.width = config["width"]
        self.act = getActivation(config)
        self.bn_mode = config.get("bn_mode", "per_layer")
        self.fc_module = nn.ModuleList()
        if self.bn_mode == "per_layer":
            self.bn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.fc_module.append(nn.Linear(in_features, self.width))
            else:
                self.fc_module.append(nn.Linear(self.width, self.width))
            if self.bn_mode == "per_layer":
                self.bn_layers.append(nn.BatchNorm1d(self.width))
    
    def forward(self, x):
        out = x
        if self.bn_mode == "per_layer":
            for layer, bn in zip(self.fc_module, self.bn_layers):
                out = self.act(bn(layer(out)))
        else:  # "per_block" or no BN
            for layer in self.fc_module:
                out = self.act(layer(out))
        return out

class DenseNetRegression(nn.Module):
    """
    DenseNet-style regression network.
    
    The network applies an initial projection to a fixed width, then uses a series of dense modules.
    Each module's output is concatenated with its input, increasing feature dimensions, and then BN (if per_block) 
    and dropout are applied per block. Finally, a linear layer maps the features to output targets.
    
    Config keys:
      - "input_shape": input feature dimension.
      - "width": width for initial projection and each module's output.
      - "depth": number of dense modules.
      - "skip_block_layers": number of linear layers per module.
      - "n_targets": output dimension.
      - "activation": activation function.
      - "bn_mode": either "per_layer" or "per_block".
      - "dropout_rate": dropout rate per block.
    """
    def __init__(self, config):
        super(DenseNetRegression, self).__init__()
        self.input_shape = config["input_shape"]
        self.width = config["width"]
        self.depth = config["depth"]
        self.n_targets = config["n_targets"]
        self.act = getActivation(config)
        self.bn_mode = config.get("bn_mode", "per_layer")
        self.dropout_rate = config.get("dropout_rate", 0.0)

        # Input projection
        self.input_layer = nn.Linear(self.input_shape, self.width)
        
        # Build dense modules.
        modules = []
        self.bn_blocks = nn.ModuleList() if self.bn_mode == "per_block" else None
        self.dropouts = nn.ModuleList() if self.dropout_rate > 0 else None
        current_features = self.width  # features after input projection
        for _ in range(self.depth):
            module = DenseModule(config, in_features=current_features)
            modules.append(module)
            current_features = current_features + self.width  # concat module output
            if self.bn_mode == "per_block":
                self.bn_blocks.append(nn.BatchNorm1d(current_features))
            if self.dropout_rate > 0:
                self.dropouts.append(nn.Dropout(self.dropout_rate))
        self.dense_modules = nn.ModuleList(modules)
        
        # Final output layer
        self.output_layer = nn.Linear(current_features, self.n_targets)
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i, module in enumerate(self.dense_modules):
            new_features = module(x)
            x = torch.cat([x, new_features], dim=1)
            if self.bn_mode == "per_block":
                x = self.act(self.bn_blocks[i](x))
            if self.dropout_rate > 0:
                x = self.dropouts[i](x)
        x = self.output_layer(x)
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
    #summary = pms.summary(regressor, torch.zeros((config["input_shape"],)).to(get_device()).double().clone().detach().requires_grad_(True)).rstrip().split('\n')
    dummy_input = torch.zeros((1, config["input_shape"])).to(get_device()).double().requires_grad_(True)
    summary = pms.summary(regressor, dummy_input).rstrip().split('\n')

    config["trainable_parameters"] = int(summary[-3].replace(',', '')[18:])
    config["non_trainable_parameters"] = int(summary[-2].replace(',', '')[22:])
    config["total_parameters"] = int(summary[-4].replace(',', '')[14:])
    
    # save config
    # with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
    #     json.dump(config, f, indent=4)
        
    return regressor

