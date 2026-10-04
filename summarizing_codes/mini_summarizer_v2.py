import os, json, re
import pandas as pd
from pathlib import Path

MODELS_DIR = "models"
OUT_XLSX = "skipmini_results.xlsx"
EMBED_IMAGES = True
COLLECT_REAL_IMAG = True   # ← turn ON to map y{odd}=y_real, y{even}=y_imag

SCALED_NAME = "hist_scaled_overlay.png"
UNSCALED_NAME = "hist_unscaled_overlay.png"
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"

def flatten(d, parent="", sep="_"):
    out = {}
    for k, v in d.items():
        nk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, nk, sep))
        else:
            out[nk] = v
    return out

_num = lambda s: int(re.findall(r"\d+", s)[0]) if re.findall(r"\d+", s) else 0

def remap_per_target(per_target: dict) -> dict:
    """Map y1→y_real, y2→y_imag, y3→y_real_2, y4→y_imag_2, ..."""
    if not isinstance(per_target, dict):
        return per_target
    out = {}
    keys = sorted(per_target.keys(), key=_num)
    for i, k in enumerate(keys, start=1):
        pair_idx = (i + 1) // 2  # 1,1,2,2,3,3,...
        base = "y_real" if i % 2 == 1 else "y_imag"
        name = base if pair_idx == 1 else f"{base}_{pair_idx}"
        out[name] = per_target[k]
    return out

rows = []
for folder in sorted(os.listdir(MODELS_DIR)):
    if not folder.startswith("skip-mini"):
        continue
    p = Path(MODELS_DIR) / folder
    if not p.is_dir():
        continue

    cfg_p, met_p = p / CONFIG_NAME, p / METRICS_NAME
    if not met_p.exists():
        continue  # only collect when metrics.json exists

    cfg = {}
    if cfg_p.exists():
        try:
            cfg = json.loads(cfg_p.read_text())
        except Exception as e:
            print(f"[WARN] Bad config in {folder}: {e}")

    try:
        met = json.loads(met_p.read_text())
    except Exception as e:
        print(f"[WARN] Bad metrics in {folder}: {e}")
        continue

    # ---- REMAP METRICS PER-TARGET (optional) ----
    if COLLECT_REAL_IMAG and isinstance(met, dict):
        for scale in ("scaled", "unscaled"):
            pt = met.get(scale, {}).get("per_target")
            if isinstance(pt, dict):
                met[scale]["per_target"] = remap_per_target(pt)

    row = {
        "model_folder": folder,
        "scaled_plot": str(p / SCALED_NAME) if (p / SCALED_NAME).exists() else "",
        "unscaled_plot": str(p / UNSCALED_NAME) if (p / UNSCALED_NAME).exists() else "",
    }
    row.update({f"config_{k}": v for k, v in flatten(cfg).items()})
    row.update({f"metrics_{k}": v for k, v in flatten(met).items()})
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    print("No eligible skip-mini folders with metrics.json found.")
else:
    if EMBED_IMAGES:
        import xlsxwriter  # ensure installed
        with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
            df.to_excel(xw, sheet_name="results", index=False)
            ws = xw.sheets["results"]
            cols = {c: i for i, c in enumerate(df.columns)}
            img_opts = {"x_scale": 0.25, "y_scale": 0.25}
            for r, row in enumerate(df.itertuples(index=False), start=1):
                for col in ("scaled_plot", "unscaled_plot"):
                    path = getattr(row, col)
                    if path and Path(path).exists():
                        ws.insert_image(r, cols[col], path, img_opts)
            for j, col in enumerate(df.columns):
                width = min(60, max(12, int(df[col].astype(str).map(len).max()) + 2))
                ws.set_column(j, j, width)
    else:
        # no embedding, smaller file
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
            df.to_excel(xw, index=False)

    print(f"✅ Wrote {OUT_XLSX} with {len(df)} rows.")
