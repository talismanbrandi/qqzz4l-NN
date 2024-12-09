import re
import pandas as pd
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark.sql.types import StructType, StructField, DoubleType
import time
import logging


def get_device():
    ''' function to get the device the NN is running on, CPU or GPU
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")
        
    return device


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
             .config('spark.driver.maxResultSize', '20g')
             .config('spark.debug.maxToStringFields', 100)
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

    # create a schema to hold the input data
    header = input_var + target_var
    schema = StructType([StructField(header[i], DoubleType(), True) for i in range(config['input_shape']+config['amplitude_shape'])])
    #print(schema)
    # load train data
    df = {}
    df['train'] = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+'train/*.csv.*', header='true')
    
    if config['var_y'] == 'all': config['var_y'] = df['train'].columns[config['input_shape']:] # this line selects all the y values
    
    # decides whether to use the monte carlo (MC) data for test and eval set
    if config['use_MC_sample']:
        mc_suffix = '_mc'
    else:
        mc_suffix = ''
    
    # load eval and test set
    df['validate'] = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+'validate'+mc_suffix+'/*.csv.*', header='true')
    
    df['test'] = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+'test'+mc_suffix+'/*.csv.*', header='true')

    logging.info(' data loaded into Spark session in {:.3f} seconds'.format(time.time() - start))
    
    # start the transfer the data to a pandas dataframe
    start = time.time()
    train_sample = min(config['train-sample-size'], df['train'].count())
    validate_sample = min(config['validate-sample-size'], df['validate'].count())
    test_sample = min(config['test-sample-size'], df['test'].count())
    
    # Select the y values to be predicted from the dataset.
    ##### Changes to support Arithmetic operations ######
    # these changes were added for supporting the prediction of arithmetic expression between two y values
    # config['var_y'] is the list containing the required fields (can be an individual y or an expression with y values)
    # the following has the supported operations
    operators = ['+', '-', '*', '/']

    # Separate the items with and without operators
    # (e.g: if we have items [y1, y2, y3+y4], we create two lists [y1, y2] and [y3+y4])
    # List of items with operators
    with_operator = [item for item in config['var_y'] if any(op in item for op in operators)]
    print(with_operator)
    # List of items without operators
    without_operator = [item for item in config['var_y'] if all(op not in item for op in operators)]
    print(without_operator)
    
    # Split the items with operators into individual components
    split_items = [subitem for item in with_operator for subitem in re.split(r'\+|\-|\*|\/', item)]

    # Combine the split items with the items without operators (might contain duplicates)
    combined_list = without_operator + split_items

    # Remove duplicates to get a unique list
    unique_list = list(set(combined_list))

    # Sort the list if you want it in a specific order (optional, but improves readability)
    unique_list.sort()
    
    df['train'] = df['train'].select(*input_var, *unique_list).limit(train_sample).toPandas() 
    df['validate'] = df['validate'].select(*input_var, *unique_list).limit(validate_sample).toPandas()
    df['test'] = df['test'].select(*input_var, *unique_list).limit(test_sample).toPandas()
    
    # For each part of the dataset, apply arithmetic operations if any
    for key in ['train', 'validate', 'test']:
        for expr in config['var_y']: # check each item to be predicted
            # Check if the "expr" item is an arithmetic expression
            if any(op in expr for op in ['+', '-', '*', '/']):
                # Split the expression into individual components and operator
                components = re.split(r'(\+|\-|\*|\/)', expr)

                # Evaluate the expression and create a new column
                if len(components) == 3:  # This should match the pattern "y1 + y3"
                    col1, operator, col2 = components

                    if operator == '+':
                        df[key][expr] = df[key][col1] + df[key][col2]
                    elif operator == '-':
                        df[key][expr] = df[key][col1] - df[key][col2]
                    elif operator == '*':
                        df[key][expr] = df[key][col1] * df[key][col2]
                    elif operator == '/':
                        df[key][expr] = df[key][col1] / df[key][col2]

    # Filter split_items to keep only those that are also in without_operator
    # The filtered_items would contain the y items that are not specified explicitly in the input 
    filtered_items = [item for item in split_items if item not in without_operator]

    # Drop the filtered items from the DataFrames
    for key in ['train', 'validate', 'test']:
        df[key] = df[key].drop(columns=filtered_items, errors='ignore')

    # print the loaded data size
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
    """ a function to normalize the target distribution
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
        # if config['var_y'] == 'all':
        #     y = df.iloc[:, config['input_shape']:]
        # else:
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


if __name__ == '__main__':
        
    config = {
        "model_type": "skip-light",
        "input_shape": 4,
        "amplitude_shape": 36,
        "data_dir": "/scratch/akula.ha/dataset", 
        "seed": 42,
        "var_y": ["y5", "y6"],
        "activation": "swish", 
        "width": 32, 
        "depth": 16, 
        "skip_block_layers": 3, 
        "beta": 0,
        "alpha": 0,
        "normal_scaled": False,
        "lr_decay_type": "lambda_exp",
        "initial_lr": 0.001,
        "final_lr": 1e-06,
        "decay_steps": 120000, 
        "train-sample-size": 5000000,
        "validate-sample-size": 500000,
        "test-sample-size": 500000,
        "use_MC_sample": False,
        "batch_size": 512,
        "steps_per_epoch": 2400,
        "early_stopping_start_epoch": 50, 
        "patience": 150, 
        "monitor": "val_mse",
        "loss": "mse",
        "gradient_clipping": True,
        "verbose": 1,
        "base_directory": "../models/",
        "epochs": 2000, 
        "model-uuid": "UUID"
    }

    load_data(config)

