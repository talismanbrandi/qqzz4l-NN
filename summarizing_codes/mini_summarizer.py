import os
import re
import json
import pandas as pd

def flatten_json(y, parent_key='', sep='_'):
    """Recursively flattens a nested json."""
    items = []
    for k, v in y.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_metrics_file(file_path, var_y=None):
    """
    Parse metrics.txt file for skip-mini folders.
    Converts it to a structure similar to test_metrics in config.
    """
    metrics = {"r2": {}, "abs_score": {}}
    var_y = var_y or []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Overall R2"):
                metrics["r2"]["model"] = float(line.split(":")[1])
            elif line.startswith("Overall abs_score"):
                metrics["abs_score"]["model"] = float(line.split(":")[1])
            elif line.startswith("Y["):
                # Example: Y[0] - MAPE: ... | R2: ... | abs_score: ...
                parts = line.split('|')
                y_index = int(line.split(']')[0][2:])
                y_key = var_y[y_index] if y_index < len(var_y) else f"y{y_index}"

                for part in parts:
                    part = part.strip()
                    if part.startswith("R2:"):
                        metrics["r2"][y_key] = float(part.split(":")[1])
                    elif part.startswith("abs_score:"):
                        metrics["abs_score"][y_key] = float(part.split(":")[1])
    return metrics

models_dir = "models"
all_configs = []
untrained_models = []

for model_folder in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_folder)
    if not os.path.isdir(model_path):
        continue

    is_skip_mini = model_folder.startswith("skip-mini")
    model_uuid = None

    if is_skip_mini:
        # Extract uuid from pattern skip-mini_uuid_num_num_num
        match = re.match(r"skip-mini_([^_]+)_\d+_\d+_\d+", model_folder)
        if match:
            model_uuid = match.group(1)

    config_path = os.path.join(model_path, "config.json")
    metrics_path = os.path.join(model_path, "metrics.txt")

    config_json = None
    metrics_json = None

    # Load config if present
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_json = json.load(f)
        if model_uuid:  # Add model_uuid for skip-mini
            config_json["model_uuid"] = model_uuid
        else:
            config_json["model_name"] = model_folder

    # Load metrics depending on type
    if is_skip_mini and os.path.exists(metrics_path):
        var_y = config_json.get("var_y") if config_json else None
        metrics_json = parse_metrics_file(metrics_path, var_y)
    elif (not is_skip_mini) and config_json and "test_metrics" in config_json:
        metrics_json = config_json.get("test_metrics")

    # Combine config and metrics if both exist
    if config_json and metrics_json:
        config_json["test_metrics"] = metrics_json
        flat_config = flatten_json(config_json)
        all_configs.append(flat_config)
    else:
        # Only skip-mini folders without either config or metrics go to untrained
        if is_skip_mini:
            untrained_models.append({"model_uuid": model_uuid or model_folder})

# Convert to DataFrames
trained_df = pd.DataFrame(all_configs)
untrained_df = pd.DataFrame(untrained_models)

# Save to Excel
trained_df.to_excel("trained_mini.xlsx", index=False)
untrained_df.to_excel("untrained_mini.xlsx", index=False)

print("✅ Created 'trained_mini.xlsx' and 'untrained_mini.xlsx' (skip-mini only for untrained).")
