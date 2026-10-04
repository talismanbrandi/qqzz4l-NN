# %%
import numpy as np
import pandas as pd
from glob import glob
from sys import exit
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import time
import os
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from datetime import datetime
import uuid
from supporting.load_data import load_data, build_data_loaders, x_scale, y_scale, y_unscale
from tqdm import tqdm
import json

# %%
config = {
    "model_type": "skip-mini",
    "input_shape": 4,
    "amplitude_shape": 36,
    "data_dir": "/scratch/hakula/dataset",
    "seed": 42,
    "var_y": [
        "y5",
        "y6"
    ],
    "use_MC_sample": False,
    "train-sample-size": 10000000,
    "validate-sample-size": 500000,
    "test-sample-size": 500000,
    "activation": "leaky_relu",
    "width": 30,
    "depth": 12,
    "skip_block_layers": 5,
    # "beta": 0,
    # "alpha": 0,
    # "normal_scaled": False,
    "lr_decay_type": "plateau",
    "initial_lr": 0.001,
    # "final_lr": 1e-06,
    "lr_patience": 15,
    "lr_factor":0.85,
    "lr_delta":7e-5,
    # "decay_steps": 200,
    "batch_size": 1024,
    "steps_per_epoch": 2400,
    "early_stopping_start_epoch": 150,
    "patience": 25,
    # "monitor": "val_mse",
    # "loss": "mse",
    # "gradient_clipping": True,
    # "verbose": 1,
    "base_directory": "../models/",
    "epochs": 2000,
    "model-uuid": "UUID"
}

# %%
torch.manual_seed(config['seed'])
torch.set_default_dtype(torch.float64)

# %%
df, spark = load_data(config)

# %%
x_cols = ['x1', 'x2', 'x3', 'x4']
y_cols = config["var_y"]
x_train = df["train"][x_cols]
y_train = df["train"][y_cols]
x_val = df["validate"][x_cols]
y_val = df["validate"][y_cols]
x_test = df["test"][x_cols]
y_test = df["test"][y_cols]

# %%
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
    ''' function for scaling y1
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



# %%
def get_device():
    ''' function to get the device the NN is running on, CPU or GPU
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")
        
    return device


# %%
class MyDataset(torch.utils.data.Dataset):
    '''Class to convert dataframe to torch tensor'''
    def __init__(self, x, y, config):
        x = x.copy(deep=True)
        y = y.copy(deep=True)

        # Vectorized scaling (if applicable)
        x['x1'] = x_scale(x['x1'])  # Or use apply if x_scale isn't vectorized

        # Number of targets
        config['n_targets'] = y.shape[1]

        for column in y.columns:
            y[column] = y[column].apply(y_scale)

        device = get_device()
        self.x = torch.tensor(x.values, dtype=torch.float64).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float64).reshape(-1, config['n_targets']).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# %%
train_dataset = MyDataset(x_train, y_train, config)
val_dataset   = MyDataset(x_val, y_val, config)
test_dataset  = MyDataset(x_test, y_test, config)

# load the data into torch tensor batches
batch_size = config["batch_size"]
numworkers = 4 if get_device().type == 'cpu' else 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=numworkers, drop_last=True)


# %%
def get_activation(activation_str):
    if activation_str == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_str == 'relu':
        return nn.ReLU()
    if activation_str == 'softplus':
        return nn.Softplus()
    if activation_str == 'swish':
        return nn.SiLU()
    if activation_str == 'sigmoid':
        return nn.Sigmoid()
    if activation_str == 'tanh':
        return nn.Tanh()
    if activation_str == 'prelu':
        return nn.PReLU()
    if activation_str == 'elu':
        return nn.ELU()
        

# %%
# Residual module (equivalent to 'module' in Keras)
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, width, n_skip_layers, activation='swish'):
        super().__init__()
        layers = []
        orig_in_dim = in_dim
        for _ in range(n_skip_layers - 1):
            layers.append(nn.Linear(in_dim, width))
            layers.append(get_activation(activation))
            in_dim = width
        layers.append(nn.Linear(width, orig_in_dim))  # projection back to input dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)

# %%
# Full model (equivalent to 'model' in Keras)
class SkipModel(nn.Module):
    def __init__(self, input_dim, width, depth, n_skip_layers, output_dim, activation='swish'):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            get_activation(activation)
        )
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(in_dim=width, width=width, n_skip_layers=n_skip_layers, activation=activation) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


# %%
# Scheduler helper
def get_scheduler(optimizer, scheduler_type, steps_per_epoch, config):
    if scheduler_type == "exponential":
        decay_rate = config["decay_rate"]
        decay_steps = config["decay_steps"]
        lr_lambda = lambda epoch: decay_rate ** (epoch / decay_steps)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif scheduler_type == "plateau":
        scheduler_args = {
            "optimizer": optimizer,
            "mode": "min",
            "factor": config["lr_factor"],
            "patience": config["lr_patience"],
            "threshold": config["lr_delta"],
            "verbose": True
        }

        # Only add min_lr if final_lr is explicitly provided
        if "final_lr" in config:
            scheduler_args["min_lr"] = config["final_lr"]

        return optim.lr_scheduler.ReduceLROnPlateau(**scheduler_args)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

# %%
# Training loop

def run_training(model, train_loader, val_loader, config, optimizer, scheduler, output_dir, uuid_str):
    device = get_device()
    model.to(device)

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []

        #print(f"Training epoch: {epoch}...")
        #for xb, yb in tqdm(train_loader):

        for xb, yb in (train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses, mapes, r2s = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss = criterion(pred, yb)
                val_losses.append(vloss.item())
                mapes.append(mean_absolute_percentage_error(yb.cpu(), pred.cpu()))
                r2s.append(r2_score(yb.cpu(), pred.cpu()))

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_mape = np.mean(mapes)
        abs_score = (1 - avg_mape) * 100
        avg_r2 = np.mean(r2s)
        lr = optimizer.param_groups[0]['lr']
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{now}] Epoch {epoch} | LR: {lr:.5e} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | Val MAPE: {avg_mape:.6f} | Val R2: {avg_r2:.6f} | abs_score: {abs_score:.4f}")

        if config['lr_decay_type'] == 'plateau':
            scheduler.step(avg_val)
        else:
            scheduler.step()

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(output_dir, f"{uuid_str}.latest.pt"))

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, f"{uuid_str}.best_weights.pt"))

        if epoch >= config['early_stopping_start_epoch']:
            if avg_val < best_val_loss:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= config['patience']:
                    print("Early stopping triggered.")
                    break


# %%
def evaluate_model(model, dataloader, device, output_dir):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dataloader):
            xb = xb.to(device)
            pred = model(xb).cpu()
            all_preds.append(pred)
            all_targets.append(yb.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    overall_mape = mean_absolute_percentage_error(y_true, y_pred)
    overall_r2 = r2_score(y_true, y_pred)
    abs_score = (1 - overall_mape) * 100

    print("\n=== Evaluation ===")
    print(f"Overall MAPE      : {overall_mape:.6f}")
    print(f"Overall R2        : {overall_r2:.6f}")
    print(f"Overall abs_score : {abs_score:.4f}")
    
    for i in range(y_true.shape[1]):
        mape_i = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        abs_score_i = (1 - mape_i) * 100
        print(f"Y[{i}] - MAPE: {mape_i:.6f} | R2: {r2_i:.6f} | abs_score: {abs_score_i:.4f}")

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Overall MAPE      : {overall_mape:.6f}\n")
        f.write(f"Overall R2        : {overall_r2:.6f}\n")
        f.write(f"Overall abs_score : {abs_score:.4f}\n")
        for i in range(y_true.shape[1]):
            mape_i = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])
            r2_i = r2_score(y_true[:, i], y_pred[:, i])
            abs_score_i = (1 - mape_i) * 100
            f.write(f"Y[{i}] - MAPE: {mape_i:.6f} | R2: {r2_i:.6f} | abs_score: {abs_score_i:.4f}\n")


# %%
if config["model-uuid"] == "UUID":
    uuid_str = str(uuid.uuid4())[:8]
else:
    uuid_str = config["model-uuid"]

# %%
# model params
width = config['width']
depth = config['depth']
n_skip_layers = config['skip_block_layers']
activation = config['activation']

# training params
steps_per_epoch = config['steps_per_epoch']
max_epochs = config['epochs']
every_n_epochs = 250
patience = config['patience']
early_stop_start = config['early_stopping_start_epoch']

# %%
model = SkipModel(config["input_shape"], width, depth, n_skip_layers, len(y_cols), activation)
optimizer = optim.Adam(model.parameters(), lr=config['initial_lr'])
scheduler = get_scheduler(optimizer, config['lr_decay_type'], config['steps_per_epoch'], config)

# %%
output_dir = os.path.join(config['base_directory'], f"{config['model_type']}_{uuid_str}_{config['width']}_{config['depth']}_{config['skip_block_layers']}")
os.makedirs(output_dir, exist_ok=True)
# Save config dictionary to output directory
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# %%
print(f"Starting training for model: {uuid_str}")
run_training(model, train_loader, val_loader, config, optimizer, scheduler, output_dir, uuid_str)

evaluate_model(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), output_dir=output_dir)



