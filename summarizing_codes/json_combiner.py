import json
import pandas as pd

def flatten_json(y, parent_key='', sep='_'):
    """
    Recursively flattens nested dicts and lists into key paths for Excel columns.
    """
    items = []
    if isinstance(y, list):
        for i, v in enumerate(y):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.extend(flatten_json(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(flatten_json(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    else:
        items.append((parent_key, y))
    return dict(items)

# Detect all raw JSONs in current directory
import glob
raw_files = glob.glob("raw*.json")
print(f"Found raw files: {raw_files}")

all_entries = []
for file in raw_files:
    with open(file, 'r') as f:
        all_entries.extend(json.load(f))

trained_data = []
untrained_skip = []
untrained_non_skip = []

for entry in all_entries:
    cfg = entry["config_json"]
    metrics = entry["metrics_json"]
    skip_mini = entry["is_skip_mini"]

    if cfg and metrics:
        cfg["model_uuid"] = entry["model_uuid"]
        cfg["model_folder"] = entry["model_folder"]
        cfg["is_skip_mini"] = skip_mini
        cfg["test_metrics"] = metrics
        trained_data.append(flatten_json(cfg))
    else:
        untrained_entry = {
            "model_uuid": entry["model_uuid"],
            "model_folder": entry["model_folder"],
            "missing": ",".join(
                k for k, v in {
                    "config": entry["has_config"],
                    "metrics": entry["has_metrics"]
                }.items() if not v
            )
        }
        if skip_mini:
            untrained_skip.append(untrained_entry)
        else:
            untrained_non_skip.append(untrained_entry)

trained_df = pd.DataFrame(trained_data)
untrained_skip_df = pd.DataFrame(untrained_skip)
untrained_non_skip_df = pd.DataFrame(untrained_non_skip)

with pd.ExcelWriter("models_summary.xlsx") as writer:
    trained_df.to_excel(writer, sheet_name="trained_all", index=False)
    untrained_skip_df.to_excel(writer, sheet_name="untrained_skip_mini", index=False)
    untrained_non_skip_df.to_excel(writer, sheet_name="untrained_non_skip_mini", index=False)

print("✅ Created models_summary.xlsx with sheets:")
print("   - trained_all")
print("   - untrained_skip_mini")
print("   - untrained_non_skip_mini")
