import os
import re
import json

def parse_metrics_file(file_path, var_y=None):
    """
    Parse metrics.txt for skip-mini folders and convert to test_metrics-like dict.
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
                parts = line.split('|')
                y_index = int(line.split(']')[0][2:])
                y_key = var_y[y_index] if var_y and y_index < len(var_y) else f"y{y_index}"
                for part in parts:
                    part = part.strip()
                    if part.startswith("R2:"):
                        metrics["r2"][y_key] = float(part.split(":")[1])
                    elif part.startswith("abs_score:"):
                        metrics["abs_score"][y_key] = float(part.split(":")[1])
    return metrics

models_dir = "models"
collected_data = []

for model_folder in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_folder)
    if not os.path.isdir(model_path):
        continue

    is_skip_mini = model_folder.startswith("skip-mini")
    model_uuid = None

    if is_skip_mini:
        match = re.match(r"skip-mini_([^_]+)_\d+_\d+_\d+", model_folder)
        if match:
            model_uuid = match.group(1)

    # Paths for config & metrics
    config_path = os.path.join(model_path, "config.json") if is_skip_mini else None
    metrics_path = os.path.join(model_path, "metrics.txt") if is_skip_mini else None

    # Non-skip-mini: pick first config*.json
    if not is_skip_mini:
        config_candidates = [f for f in os.listdir(model_path) if f.startswith("config") and f.endswith(".json")]
        config_candidates.sort()
        config_path = os.path.join(model_path, config_candidates[0]) if config_candidates else None

    has_config = config_path and os.path.exists(config_path)
    has_metrics = metrics_path and os.path.exists(metrics_path)
    config_json = None
    metrics_json = None

    if has_config:
        with open(config_path, 'r') as f:
            config_json = json.load(f)

    # Parse metrics differently
    if is_skip_mini and has_metrics:
        var_y = config_json.get("var_y") if config_json else None
        metrics_json = parse_metrics_file(metrics_path, var_y)
    elif (not is_skip_mini) and config_json and "test_metrics" in config_json:
        metrics_json = config_json.get("test_metrics")

    # Store raw info
    collected_data.append({
        "model_folder": model_folder,
        "model_uuid": model_uuid or model_folder,
        "is_skip_mini": is_skip_mini,
        "has_config": has_config,
        "has_metrics": has_metrics,
        "config_json": config_json,
        "metrics_json": metrics_json
    })

# Save raw JSON
output_file = "raw_model_data.json"  # rename per system if needed
with open(output_file, 'w') as f:
    json.dump(collected_data, f, indent=2)

print(f"✅ Saved raw model data to {output_file}")
