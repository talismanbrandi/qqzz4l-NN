import os
import json
import pandas as pd
from tqdm import tqdm

# Replace this with your actual path
BASE_DIR = "../models"

# Output lists
complete_configs = []
incomplete_configs = []

# Traverse all subdirectories
for subdir, _, files in tqdm(os.walk(BASE_DIR)):
    config_path = None
    has_test_results = any(f.startswith("test-results") and f.endswith(".csv") for f in files)

    for f in files:
        if f.startswith("config") and f.endswith(".json"):
            config_path = os.path.join(subdir, f)
            break

    if config_path:
        try:
            with open(config_path, 'r') as json_file:
                config_data = json.load(json_file)
        except Exception as e:
            print(f"Error reading {config_path}: {e}")
            continue

        config_data["training_status"] = "complete" if has_test_results else "incomplete"
        config_data["config_path"] = config_path

        if has_test_results:
            complete_configs.append(config_data)
        else:
            incomplete_configs.append(config_data)

# Convert to DataFrames
df_complete = pd.DataFrame(complete_configs)
df_incomplete = pd.DataFrame(incomplete_configs)

# Expand the 'test_metrics' dictionary into individual columns
def flatten_metrics(row):
    metrics = row.get("test_metrics", {})
    flat = {}
    for outer_key, inner_dict in metrics.items():
        for inner_key, value in inner_dict.items():
            flat[f"tm_{outer_key}_{inner_key}"] = value  # << Prefix added here
    return pd.Series(flat)

# Apply and merge
if "test_metrics" in df_complete.columns:
    test_metrics_expanded = df_complete.apply(flatten_metrics, axis=1)
    df_complete = pd.concat([df_complete.drop(columns=["test_metrics"]), test_metrics_expanded], axis=1)

# Save results for inspection
df_complete.to_csv("complete_configs.csv", index=False)
df_incomplete.to_csv("incomplete_configs.csv", index=False)

# Display summary
print("Complete Trainings:", len(df_complete))
print("Incomplete Trainings:", len(df_incomplete))
print("All unique config keys:", set().union(*(d.keys() for d in complete_configs + incomplete_configs)))
