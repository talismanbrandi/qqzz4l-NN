#!/usr/bin/env python3
"""Download per-run inference CSVs from the model and ensemble experiments.

Run from the repository root:
    python scripts/utils/download_inference_results.py

Uses ./secrets/.databrickscfg by default. Override with
DATABRICKS_CONFIG_FILE or --databricks-config. Existing output files are skipped.
Each run's full artifact folder is downloaded temporarily; only its renamed CSV
is retained in the output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from tempfile import TemporaryDirectory


EXPERIMENT_PREFIX = "/Users/harish.akula096@gmail.com/"
ARTIFACT_NAME = "inference-results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-experiment",
        default=EXPERIMENT_PREFIX + "regressor-reevaluate-stricter-1e-5-v4",
        help="Full MLflow experiment name for individual models.",
    )
    parser.add_argument(
        "--ensembles-experiment",
        default=EXPERIMENT_PREFIX + "regressor-ensembles-stricter-1e-5-v4",
        help="Full MLflow experiment name for ensembles.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
        help="MLflow server URI (default: MLFLOW_TRACKING_URI or databricks).",
    )
    parser.add_argument(
        "--databricks-config",
        default=os.getenv("DATABRICKS_CONFIG_FILE", "./secrets/.databrickscfg"),
        help="Databricks credentials file; contents are never printed.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inference-results"))
    parser.add_argument(
        "--artifact-path",
        help="CSV path within the downloaded run folder; otherwise search it recursively.",
    )
    parser.add_argument(
        "--include-deleted", action="store_true", help="Also search deleted runs."
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    """Produce a portable basename, keeping room for a run ID suffix."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:160] or "run"


def list_runs(client, experiment_id: str, view_type: int) -> list:
    runs = []
    token = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            run_view_type=view_type,
            max_results=1000,
            page_token=token,
        )
        runs.extend(page)
        token = page.token
        if not token:
            return sorted(runs, key=lambda run: run.info.run_id)


def find_artifact_paths(artifact_root: Path, artifact_path: str | None) -> list[Path]:
    """Locate the CSV in a downloaded run folder, without more server requests."""
    if artifact_path:
        relative = PurePosixPath(artifact_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("--artifact-path must be a run-relative path without '..'.")
        candidates = [artifact_root.joinpath(*relative.parts)]
    else:
        candidates = sorted(artifact_root.rglob(ARTIFACT_NAME))
    root = artifact_root.resolve()
    return [
        path for path in candidates
        if path.is_file() and path.resolve().is_relative_to(root)
    ]


def download_experiment(client, experiment_name, output_dir, view_type, artifact_path):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    runs = list_runs(client, experiment.experiment_id, view_type)
    if not runs:
        raise ValueError(f"No runs found in experiment: {experiment_name}")

    names = [safe_name(run.data.tags.get("mlflow.runName") or run.info.run_id) for run in runs]
    counts = Counter(name.casefold() for name in names)
    used = set()
    downloaded = skipped = missing = failed = 0
    print(f"{experiment_name}: {len(runs)} runs", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for run, name in zip(runs, names):
        run_id = run.info.run_id
        if counts[name.casefold()] > 1:
            name = f"{name}__{run_id}"
        # Also handle a run name that resembles another run's suffixed name.
        while name.casefold() in used:
            name += f"__{run_id}"
        used.add(name.casefold())
        destination = output_dir / f"{name}.csv"
        if destination.exists():
            print(f"  SKIP {destination}", flush=True)
            skipped += 1
            continue

        try:
            print(f"  DOWNLOADING all artifacts: {name} ({run_id})", flush=True)
            # Stage the full run folder so interrupted transfers never leave a final CSV.
            with TemporaryDirectory(prefix=".download-", dir=output_dir) as temporary:
                artifact_root = Path(client.download_artifacts(
                    run_id, path="", dst_path=temporary
                ))
                paths = find_artifact_paths(artifact_root, artifact_path)
                if not paths:
                    print(
                        f"  MISSING {name} ({run_id}): {artifact_path or ARTIFACT_NAME}",
                        file=sys.stderr,
                    )
                    missing += 1
                    continue
                if len(paths) != 1:
                    relative_paths = [str(path.relative_to(artifact_root)) for path in paths]
                    raise ValueError(
                        f"Multiple matching artifacts: {relative_paths}. "
                        "Select one with --artifact-path."
                    )
                staged = Path(temporary) / ".inference-result-staged.csv"
                shutil.copyfile(paths[0], staged)
                # Linking publishes the complete file atomically without overwriting.
                os.link(staged, destination)
            downloaded += 1
            print(f"  OK {destination}", flush=True)
        except FileExistsError:
            skipped += 1
            print(f"  SKIP {destination} (already exists)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  FAILED {name} ({run_id}): {exc}", file=sys.stderr)

    print(
        f"{output_dir}: {downloaded} downloaded, {skipped} skipped, "
        f"{missing} missing, {failed} failed",
        flush=True,
    )
    return missing + failed


def main() -> int:
    args = parse_args()
    try:
        import mlflow
        from mlflow.entities import ViewType
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("MLflow is required. Install the project requirements first.", file=sys.stderr)
        return 1

    os.environ["DATABRICKS_CONFIG_FILE"] = str(Path(args.databricks_config).expanduser())
    mlflow.set_tracking_uri(args.tracking_uri)
    if args.tracking_uri == "databricks" or args.tracking_uri.startswith("databricks://"):
        mlflow.login(interactive=False)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    view_type = ViewType.ALL if args.include_deleted else ViewType.ACTIVE_ONLY
    problems = 0
    for experiment_name, folder in (
        (args.models_experiment, "models"),
        (args.ensembles_experiment, "ensembles"),
    ):
        try:
            problems += download_experiment(
                client, experiment_name, args.output_dir / folder, view_type, args.artifact_path
            )
        except Exception as exc:
            problems += 1
            print(f"FAILED experiment {experiment_name}: {exc}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
