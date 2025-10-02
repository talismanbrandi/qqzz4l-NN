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
import pytorch_model_summary as pms
import argparse
import json

config = {
    "model_type": "skip-mini",
    "input_shape": 4,
    "amplitude_shape": 36,
    "data_dir": "/scratch/hakula/dataset",
    "seed": 42,
    "var_y": 'all',
    "use_MC_sample": False,
    "val_MC": True,
    "train-sample-size": 10_000_000,
    "validate-sample-size": 500_000,
    "test-sample-size": 500_000,
    "activation": "leaky_relu",
    "width": 160,
    "depth": 16,
    "skip_block_layers": 6,
    # "beta": 0,
    # "alpha": 0,
    # "normal_scaled": False,
    "lr_decay_type": "plateau",
    "initial_lr": 0.001,
    # "final_lr": 1e-6,
    "lr_patience": 25,
    "lr_factor": 0.8,
    "lr_delta": 7e-5,
    # "decay_steps": 200,
    "batch_size": 2048,
    "steps_per_epoch": 2400,
    "early_stopping_start_epoch": 100,
    "patience": 40,
    # "monitor": "val_mse",
    "loss": "huber",
    "gradient_clipping": True,
    # "verbose": 1,
    "base_directory": "../models/",
    "epochs": 2000,
    "model-uuid": "UUID",
    "device_id": 0
}

parser = argparse.ArgumentParser(description="A torch implementation of high-precision regressors",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config", help="configuration file for the run")
args = vars(parser.parse_args())

# set up the config
with open(args['config'], 'r') as f:
    config = json.load(f)

device_id = config.get("device_id", 0)
if torch.cuda.is_available():
    print(f"using device {device_id}")

torch.manual_seed(config['seed'])
torch.set_default_dtype(torch.float64)

def filter_by_y_threshold(dfs, threshold=1e-4, prefix='y', mode='all', verbose=True):
    """
    Filter rows in each DataFrame within a dict by threshold on y columns,
    and print counts before and after filtering.

    Args:
        dfs (dict): Dictionary of DataFrames (e.g., {'train': df1, 'test': df2, ...})
        threshold (float): Minimum absolute value required (default: 1e-4)
        prefix (str): Prefix for y columns (default: 'y')
        mode (str): 'all' or 'any'
        verbose (bool): Print row counts if True.

    Returns:
        dict: New dictionary with filtered DataFrames.
    """
    filtered = {}
    for key, df in dfs.items():
        y_cols = [col for col in df.columns if col.startswith(prefix)]
        n_before = len(df)
        if mode == 'all':
            mask = (df[y_cols].abs() >= threshold).all(axis=1)
        elif mode == 'any':
            mask = (df[y_cols].abs() >= threshold).any(axis=1)
        else:
            raise ValueError("mode must be 'all' or 'any'")
        filtered_df = df[mask].copy()
        n_after = len(filtered_df)
        n_dropped = n_before - n_after
        if verbose:
            print(f"{key}: {n_before} → {n_after} (dropped {n_dropped})")
        filtered[key] = filtered_df
    return filtered

def oversample_below_y_threshold(
    dfs,
    threshold=1e-4,
    prefix='y',
    mode='any',
    factor=2,
    random_state=None,
    verbose=True
):
    """
    Oversample rows where y-columns are below the threshold in each split of a DataFrame dict.

    Args:
        dfs (dict): Dictionary of DataFrames.
        threshold (float): Threshold for y-columns.
        prefix (str): Prefix for y-columns (default: 'y').
        mode (str): 'all' or 'any' (default: 'any').
        factor (int): How many times to duplicate (default: 2 = one copy added).
        random_state (int): For reproducibility.
        verbose (bool): Print counts.

    Returns:
        dict: New dictionary with oversampled DataFrames.
    """
    oversampled = {}
    for key, df in dfs.items():
        y_cols = [col for col in df.columns if col.startswith(prefix)]
        if mode == 'all':
            mask = (df[y_cols].abs() < threshold).all(axis=1)
        elif mode == 'any':
            mask = (df[y_cols].abs() < threshold).any(axis=1)
        else:
            raise ValueError("mode must be 'all' or 'any'")
        rows_to_oversample = df[mask]
        n_before = len(df)
        if len(rows_to_oversample) > 0:
            sampled = rows_to_oversample.sample(
                n=factor * len(rows_to_oversample),
                replace=True,
                random_state=random_state
            )
            df_new = pd.concat([df, sampled], ignore_index=True)
        else:
            df_new = df.copy()
        if verbose:
            n_added = len(df_new) - n_before
            print(f"{key}: {n_before} → {len(df_new)} (added {n_added})")
        oversampled[key] = df_new
    return oversampled


df, spark = load_data(config)

if config.get("use_MC_sample"):
    print("Using MC data for Test set.")
    if config.get("val_MC"):
        print("Using MC data for Val set.")

if "filter_y_threshold" in config:
    df = filter_by_y_threshold(df, threshold=config["filter_y_threshold"])

if "oversample_below_threshold" in config:
    df = oversample_below_y_threshold(
        df,
        threshold=1e-4,
        prefix='y',
        mode='any',     # or 'all' if you want stricter logic
        factor=8,       # add as many duplicates as you want
        random_state=42
    )


x_cols = ['x1', 'x2', 'x3', 'x4']
y_cols = config["var_y"]
x_train = df["train"][x_cols]
y_train = df["train"][y_cols]
x_val = df["validate"][x_cols]
y_val = df["validate"][y_cols]
x_test = df["test"][x_cols]
y_test = df["test"][y_cols]

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



def get_device():
    ''' function to get the device the NN is running on, CPU or GPU
    '''
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else: 
        device = torch.device("cpu")
        
    return device


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
    
class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff  = (y_pred - y_true).abs()
        scale = (y_pred.abs() + y_true.abs()).clamp(min=self.eps)  # avoid /0
        return torch.mean(diff / scale)

# this version is for monitoring performance, not for loss optimization
# def smape(pred, target, eps=1e-8):
#     numerator = torch.abs(pred - target)
#     denominator = (torch.abs(pred) + torch.abs(target)) / 2.0 + eps
#     return (numerator / denominator).mean()
def smape(pred, target, eps=1e-8):
    """
    SMAPE function that works for both PyTorch tensors and NumPy arrays.
    """
    # If input is numpy, convert to torch for computation
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # Ensure everything is float
    pred = pred.float()
    target = target.float()

    numerator = torch.abs(pred - target)
    denominator = (torch.abs(pred) + torch.abs(target)) / 2.0 + eps
    return (numerator / denominator).mean().item()

def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Quick factory for common regression losses.

    Parameters
    ----------
    name : {'mse', 'smape', 'huber'}
        Case-insensitive loss name.
    **kwargs
        delta (float): Huber breakpoint (default 0.01).
        eps   (float): SMAPE denominator offset (default 1e-8).
    """
    name = name.lower()

    if name == "mse":
        return nn.MSELoss()
    if name == "smape":
        return SMAPELoss(eps=kwargs.get("eps", 1e-8))
    if name == "huber":
        return nn.HuberLoss(delta=kwargs.get("delta", 0.01))
    if name == "huber+mape":
        delta = kwargs.get("delta", 0.01)
        eps = kwargs.get("eps", 1e-8)
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 1.0)
        huber = nn.HuberLoss(delta=delta)
        return lambda pred, target: alpha * huber(pred, target) + beta * (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()
    if name == "mape":
        eps = kwargs.get("eps", 1e-8)
        return lambda pred, target: (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()

    raise ValueError("name must be 'mse', 'mape', 'smape', or 'huber'")

if "monitor" in config:
    if config["monitor"] == "smape":
        mean_absolute_percentage_error = smape

train_dataset = MyDataset(x_train, y_train, config)
val_dataset   = MyDataset(x_val, y_val, config)
test_dataset  = MyDataset(x_test, y_test, config)

# load the data into torch tensor batches
batch_size = config["batch_size"]
numworkers = 4 if get_device().type == 'cpu' else 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=numworkers, drop_last=True)

        

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


# Scheduler helper
def get_scheduler(optimizer, scheduler_type, steps_per_epoch, config):
    if scheduler_type == "lambda_exp":
        decay_rate = config["decay_rate"]
        decay_steps = config["decay_steps"]
        lr_lambda = lambda epoch: decay_rate ** (epoch / decay_steps)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "exponential":
        # Standard ExponentialLR
        gamma = config.get("lr_gamma", 0.95)  # default if not provided
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == "flat_exp":
        # Custom: flat LR for warm_epochs, then exponential decay
        warm_epochs = config.get("warm_epochs", 10)
        gamma = config.get("lr_gamma", 0.95)

        def lr_lambda(epoch):
            if epoch < warm_epochs:
                return 1.0
            else:
                return gamma ** (epoch - warm_epochs)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif scheduler_type == "plateau":
        scheduler_args = {
            "optimizer": optimizer,
            "mode": "min",
            "factor": config["lr_factor"],
            "patience": config["lr_patience"],
            "threshold": config["lr_delta"]
        }

        # Only add min_lr if final_lr is explicitly provided
        if "final_lr" in config:
            scheduler_args["min_lr"] = config["final_lr"]

        return optim.lr_scheduler.ReduceLROnPlateau(**scheduler_args)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

# Training loop

import os
import shutil
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def save_checkpoint(output_dir, uuid_str, epoch, model, optimizer, scheduler,
                    best_val_loss, epochs_since_improvement, train_losses, val_losses):
    """Helper to save checkpoint with all states."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'epochs_since_improvement': epochs_since_improvement,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    latest_path = os.path.join(output_dir, f"{uuid_str}.latest.pt")
    torch.save(checkpoint, latest_path)


def load_checkpoint(checkpoint_dir, output_dir, model, optimizer, scheduler, config, uuid_str):
    """Load checkpoint if available. If keys are missing, fall back to config/defaults."""
    latest_path = None
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".latest.pt"):
            latest_path = os.path.join(checkpoint_dir, f)
            break

    if latest_path is None:
        # raise FileNotFoundError("No .latest.pt file found in checkpoint_dir")
        print("No checkpoint found. Starting training from scratch...")
        return 0, float('inf'), 0, [], []

    checkpoint = torch.load(latest_path, map_location="cpu")
    print(f"Loaded checkpoint from {latest_path}. Keys: {list(checkpoint.keys())}")

    # Restore states
    model.load_state_dict(checkpoint.get('model_state_dict', model.state_dict()))
    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
    scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))

    # Restore variables or use defaults
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    # Copy best weights if exist
    best_weights_src = None
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".best_weights.pt"):
            best_weights_src = os.path.join(checkpoint_dir, f)
            break
    if best_weights_src:
        dst = os.path.join(output_dir, f"{uuid_str}.best_weights.pt")
        try:
            shutil.copy(best_weights_src, dst)
            print(f"Copied best weights to {dst}")
        except shutil.SameFileError:
            print("Best weights already exist at destination. Skipping copy.")
        except Exception as e:
            print(f"Skipping best weights copy due to error: {e}")
    return start_epoch, best_val_loss, epochs_since_improvement, train_losses, val_losses


def run_training(model, train_loader, val_loader, config, optimizer, scheduler, output_dir, uuid_str):
    device = get_device()
    model.to(device)

    criterion = get_loss_function(config['loss'], **config.get("loss_params", {}))

    best_val_loss = float('inf')
    epochs_since_improvement = 0
    start_epoch = 0
    train_losses, val_losses = [], []

    # Load checkpoint if path provided
    if "load_checkpoint_path" in config:
        checkpoint_dir = config["load_checkpoint_path"]
        if os.path.exists(checkpoint_dir):
            start_epoch, best_val_loss, epochs_since_improvement, train_losses, val_losses = \
                load_checkpoint(checkpoint_dir, output_dir, model, optimizer, scheduler, config, uuid_str)
        else:
            raise Exception(" The provided checkpoint path does not exist!")

    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        batch_train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if config.get("gradient_clipping", False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_train_losses.append(loss.item())

        # Validation
        model.eval()
        batch_val_losses, mapes, r2s = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss = criterion(pred, yb)
                batch_val_losses.append(vloss.item())
                mapes.append(mean_absolute_percentage_error(yb.cpu(), pred.cpu()))
                r2s.append(r2_score(yb.cpu(), pred.cpu()))

        avg_train = np.mean(batch_train_losses)
        avg_val = np.mean(batch_val_losses)
        avg_mape = np.mean(mapes)
        avg_r2 = np.mean(r2s)
        abs_score = (1 - avg_mape) * 100
        lr = optimizer.param_groups[0]['lr']
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{now}] Epoch {epoch} | LR: {lr:.5e} | Train {config['loss'].upper()}: {avg_train:.6f} | "
              f"Val {config['loss'].upper()}: {avg_val:.6f} | Val MAPE: {avg_mape:.6f} | "
              f"Val R2: {avg_r2:.6f} | abs_score: {abs_score:.4f}")

        # Scheduler update
        if config['lr_decay_type'] == 'plateau':
            scheduler.step(avg_val)
        else:
            scheduler.step()

        # Append losses to history
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # Save checkpoint
        save_checkpoint(output_dir, uuid_str, epoch, model, optimizer, scheduler,
                        best_val_loss, epochs_since_improvement, train_losses, val_losses)

        # Early stopping
        improved = avg_val < (best_val_loss - 1e-12)
        if epoch >= config['early_stopping_start_epoch']:
            if improved:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= config['patience']:
                    print("Early stopping triggered.")
                    break

        if improved:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, f"{uuid_str}.best_weights.pt"))
            print(f"New best model saved at epoch {epoch} with Val {config['loss'].upper()}: {best_val_loss:.6f}")


import os, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# def _safe_mape(y, yhat):
#     mask = np.abs(y) > 1e-12
#     if mask.sum() == 0:
#         return np.nan
#     return mean_absolute_percentage_error(y[mask], yhat[mask])

def _percent_delta(y_true, y_pred):
    """(y_pred - y_true)/y_true * 100 with zero-guard → NaNs where denom≈0."""
#     delta = np.full_like(y_true, np.nan, dtype=float)
#     mask = np.abs(y_true) > 1e-12
#     delta[mask] = (y_pred[mask] - y_true[mask]) / y_true[mask] * 100.0
#     return delta
    """(y_pred - y_true)/y_true * 100 without zero-guard."""
    return (y_pred - y_true) / y_true * 100.0

def _save_error_plots_from_arrays(delta_matrix, labels, out_path, cutoff=5):
    """Overlay histogram for each target column in delta_matrix."""
    plt.figure()
    for j, name in enumerate(labels):
        col = delta_matrix[:, j]
        m = np.isfinite(col)
        if cutoff is not None:
            m &= (np.abs(col) < cutoff)
        data = col[m]
        if data.size == 0:
            continue
        plt.hist(data, bins=100, histtype='step', label=name, density=True)
    plt.grid(True, linestyle='dotted', alpha=0.5)
    plt.xlabel(r'$\delta$')
    plt.ylabel('pdf')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_losses_from_output_dir(output_dir, title="Training vs Validation Loss"):
    """
    Load the *.latest.pt checkpoint from an output directory,
    plot train/val loss curves, and save them as:
      - loss_plot.png        (linear scale)
      - loss_plot_loglog.png (log-log scale)
    """
    # Find latest.pt file
    latest_ckpt = None
    for f in os.listdir(output_dir):
        if f.endswith(".latest.pt"):
            latest_ckpt = os.path.join(output_dir, f)
            break

    if latest_ckpt is None:
        raise FileNotFoundError(f"No .latest.pt file found in {output_dir}")

    # Load checkpoint
    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    print(f"Loaded checkpoint: {latest_ckpt}")
    print(f"Available keys: {list(checkpoint.keys())}")

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])

    if not train_losses or not val_losses:
        print("No loss history found in checkpoint.")
        return

    # -------- Linear scale plot --------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved loss plot to: {save_path}")

    # -------- Log-log scale plot --------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title + " (Log-Log Scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    save_path_log = os.path.join(output_dir, "loss_plot_loglog.png")
    plt.savefig(save_path_log, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved log-log loss plot to: {save_path_log}")

    return train_losses, val_losses


def evaluate_model(model, dataloader, device, output_dir, config):
    """
    Evaluate a trained model and save:
      - metrics.json (scaled & unscaled)
      - hist_unscaled_overlay.png, hist_scaled_overlay.png
    Requires a globally available scalar y_unscale(y: float) -> float.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            all_preds.append(pred)
            all_targets.append(yb.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    y_names = config["var_y"]
    assert y_true.ndim == 2 and y_true.shape[1] == len(y_names), \
        "Mismatch between targets array shape and config['var_y']"

    # -------- Scaled metrics --------
    scaled = {}
    overall_mape = mean_absolute_percentage_error(y_true, y_pred)
    overall_r2   = r2_score(y_true, y_pred)
    overall_abs  = (1 - overall_mape) * 100 if np.isfinite(overall_mape) else np.nan
    scaled["overall"] = {"MAPE": float(overall_mape), "R2": float(overall_r2), "abs_score": float(overall_abs)}
    scaled["per_target"] = {}
    for j, name in enumerate(y_names):
        mape_j = mean_absolute_percentage_error(y_true[:, j], y_pred[:, j])
        r2_j   = r2_score(y_true[:, j], y_pred[:, j])
        abs_j  = (1 - mape_j) * 100 if np.isfinite(mape_j) else np.nan
        scaled["per_target"][name] = {"MAPE": float(mape_j), "R2": float(r2_j), "abs_score": float(abs_j)}

    # -------- Unscaled metrics --------
    # Your y_unscale is scalar-only; vectorize it.
    y_unscale_vec = np.vectorize(y_unscale)
    y_true_un = y_unscale_vec(y_true)
    y_pred_un = y_unscale_vec(y_pred)

    unscaled = {}
    overall_mape_un = mean_absolute_percentage_error(y_true_un, y_pred_un)
    overall_r2_un   = r2_score(y_true_un, y_pred_un)
    overall_abs_un  = (1 - overall_mape_un) * 100 if np.isfinite(overall_mape_un) else np.nan
    unscaled["overall"] = {"MAPE": float(overall_mape_un), "R2": float(overall_r2_un), "abs_score": float(overall_abs_un)}
    unscaled["per_target"] = {}
    for j, name in enumerate(y_names):
        mape_j_un = mean_absolute_percentage_error(y_true_un[:, j], y_pred_un[:, j])
        r2_j_un   = r2_score(y_true_un[:, j], y_pred_un[:, j])
        abs_j_un  = (1 - mape_j_un) * 100 if np.isfinite(mape_j_un) else np.nan
        unscaled["per_target"][name] = {"MAPE": float(mape_j_un), "R2": float(r2_j_un), "abs_score": float(abs_j_un)}

    # -------- Deltas (arrays, no DataFrame) --------
    scaled_delta   = _percent_delta(y_true,    y_pred)
    unscaled_delta = _percent_delta(y_true_un, y_pred_un)

    # -------- Save metrics JSON --------
    metrics = {"scaled": scaled, "unscaled": unscaled, "meta": {"num_samples": int(y_true.shape[0]), "targets": y_names}}
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # -------- Save plots (overlay style) --------
    _save_error_plots_from_arrays(
        unscaled_delta, y_names, os.path.join(output_dir, "hist_unscaled_overlay.png"), cutoff=5
    )
    _save_error_plots_from_arrays(
        scaled_delta, y_names, os.path.join(output_dir, "hist_scaled_overlay.png"), cutoff=5
    )

    # Console summary
    print("\n=== Evaluation (SCALED) ===")
    print(f"Overall MAPE      : {scaled['overall']['MAPE']:.6f}")
    print(f"Overall R2        : {scaled['overall']['R2']:.6f}")
    print(f"Overall abs_score : {scaled['overall']['abs_score']:.6f}")
    for n, v in scaled["per_target"].items():
        print(f"{n} - MAPE: {v['MAPE']:.6f} | R2: {v['R2']:.6f} | abs_score: {v['abs_score']:.6f}")

    print("\n=== Evaluation (UNSCALED) ===")
    print(f"Overall MAPE      : {unscaled['overall']['MAPE']:.6f}")
    print(f"Overall R2        : {unscaled['overall']['R2']:.6f}")
    print(f"Overall abs_score : {unscaled['overall']['abs_score']:.6f}")
    for n, v in unscaled["per_target"].items():
        print(f"{n} - MAPE: {v['MAPE']:.6f} | R2: {v['R2']:.6f} | abs_score: {v['abs_score']:.6f}")

    print(f"\nSaved metrics to: {out_path}")
    print("Saved histograms to:",
          os.path.join(output_dir, "hist_unscaled_overlay.png"), "and",
          os.path.join(output_dir, "hist_scaled_overlay.png"))
    
    plot_losses_from_output_dir(output_dir)



if config["model-uuid"] == "UUID":
    uuid_str = str(uuid.uuid4())[:8]
else:
    uuid_str = config["model-uuid"]

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

model = SkipModel(config["input_shape"], width, depth, n_skip_layers, len(y_cols), activation)
model.to(get_device())
dummy_input = torch.zeros((1, config["input_shape"])).to(get_device()).double().requires_grad_(True)
summary = pms.summary(model, dummy_input).rstrip().split('\n')
print('\n' + "\n".join(summary))


optimizer = optim.Adam(model.parameters(), lr=config['initial_lr'])
scheduler = get_scheduler(optimizer, config['lr_decay_type'], config['steps_per_epoch'], config)

output_dir = os.path.join(config['base_directory'], f"{config['model_type']}_{uuid_str}_{config['width']}_{config['depth']}_{config['skip_block_layers']}")
os.makedirs(output_dir, exist_ok=True)
# Save config dictionary to output directory
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

print(f"Starting training for model: {uuid_str}")
run_training(model, train_loader, val_loader, config, optimizer, scheduler, output_dir, uuid_str)

# Load best model weights before test set evaluation
print("Loading best weights for evaluation...")
best_weights_path = os.path.join(output_dir, f"{uuid_str}.best_weights.pt")
model.load_state_dict(torch.load(best_weights_path, map_location=get_device()))

print("Running evaluation...")
evaluate_model(model, test_loader, device=get_device(), output_dir=output_dir, config=config)


