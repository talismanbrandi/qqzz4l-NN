import os
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

models_dir = "models"
trained_configs = []
untrained_configs = []

for model_folder in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_folder)
    
    # Skip non-folders or folders starting with "skip-mini"
    if not os.path.isdir(model_path) or model_folder.startswith("skip-mini"):
        continue

    # Find config files in the folder
    config_files = [f for f in os.listdir(model_path) if f.startswith("config") and f.endswith(".json")]
    if not config_files:
        continue
    
    # Pick the first config file
    config_files.sort()
    config_file_path = os.path.join(model_path, config_files[0])

    with open(config_file_path, 'r') as f:
        config_json = json.load(f)

    # Flatten the JSON
    flat_config = flatten_json(config_json)
    flat_config["model_name"] = model_folder  # Add model name as identifier

    # Separate into trained and untrained
    if any("test_metrics" in key for key in flat_config.keys()):
        trained_configs.append(flat_config)
    else:
        untrained_configs.append(flat_config)

# Convert to DataFrame and align columns
trained_df = pd.DataFrame(trained_configs)
untrained_df = pd.DataFrame(untrained_configs)

# Save to Excel
trained_df.to_excel("trained_h100.xlsx", index=False)
untrained_df.to_excel("untrained_h100.xlsx", index=False)

print("Excel files 'trained.xlsx' and 'untrained.xlsx' have been created, ignoring 'skip-mini*' folders.")
