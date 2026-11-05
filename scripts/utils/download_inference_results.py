#!/usr/bin/env python3
"""Download inference CSVs or all artifacts from model and ensemble experiments.

Run from the repository root:
    python scripts/utils/download_inference_results.py --download-mode csv
    python scripts/utils/download_inference_results.py --download-mode all

Uses ./secrets/.databrickscfg by default. Override with
DATABRICKS_CONFIG_FILE or --databricks-config. CSV mode (default) downloads only
the selected CSV and saves it as <run_name>.csv. All mode preserves every run
artifact in <run_name>/. Existing output files or run folders are skipped.
Up to 8 runs download concurrently; set --max-workers to change this limit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from tempfile import TemporaryDirectory


EXPERIMENT_PREFIX = "/Users/harish.akula096@gmail.com/"
ARTIFACT_NAME = "inference-results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "--download-mode", choices=("csv", "all"), default="csv",
        help="Download only the inference CSV (default) or all run artifacts.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Maximum concurrent run downloads (default: 8; use 1 for sequential downloads).",
    )
    parser.add_argument(
        "--artifact-path",
        help="Run-relative CSV path (CSV mode only); otherwise search artifact listings recursively.",
    )
    parser.add_argument(
        "--include-deleted", action="store_true", help="Also search deleted runs."
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1.")
    if args.artifact_path is not None:
        if args.download_mode != "csv":
            parser.error("--artifact-path is only supported with --download-mode csv.")
        try:
            validate_artifact_path(args.artifact_path)
        except ValueError as exc:
            parser.error(str(exc))
    return args


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


def validate_artifact_path(path: str) -> str:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("--artifact-path must be a nonempty run-relative file path without '..'.")
    return relative.as_posix()


def find_artifact_paths(client, run_id: str, artifact_path: str | None) -> list[str]:
    """Find CSVs using remote metadata, without downloading unrelated artifacts."""
    if artifact_path is not None:
        target = validate_artifact_path(artifact_path)
        parent = PurePosixPath(target).parent.as_posix()
        return [
            item.path for item in client.list_artifacts(run_id, path="" if parent == "." else parent)
            if not item.is_dir and item.path == target
        ]

    matches = []
    pending = [""]
    visited = set()
    while pending:
        directory = pending.pop()
        if directory in visited:
            continue
        visited.add(directory)
        for item in client.list_artifacts(run_id, path=directory):
            if item.is_dir:
                pending.append(item.path)
            elif PurePosixPath(item.path).name == ARTIFACT_NAME:
                matches.append(item.path)
    return sorted(set(matches))


def download_run(client_factory, run_id, name, destination, artifact_path, download_mode):
    """Download one run with its own MLflow client and temporary directory."""
    if destination.exists():
        print(f"  SKIP {destination}", flush=True)
        return "skipped"

    try:
        client = client_factory()
        if download_mode == "csv":
            paths = find_artifact_paths(client, run_id, artifact_path)
            if not paths:
                print(
                    f"  MISSING {name} ({run_id}): {artifact_path or ARTIFACT_NAME}",
                    file=sys.stderr,
                )
                return "missing"
            if len(paths) != 1:
                raise ValueError(
                    f"Multiple matching artifacts: {paths}. "
                    "Select one with --artifact-path."
                )
            remote_path = paths[0]
        else:
            remote_path = ""

        label = "CSV" if download_mode == "csv" else "all artifacts"
        print(f"  DOWNLOADING {label}: {name} ({run_id})", flush=True)
        # Publish only completed downloads; failed transfers leave no final output.
        with TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
            staging_dir = Path(temporary) / "artifacts"
            staging_dir.mkdir()
            downloaded_path = Path(client.download_artifacts(
                run_id, path=remote_path, dst_path=str(staging_dir.resolve())
            ))
            if download_mode == "csv":
                staged = Path(temporary) / ".inference-result-staged.csv"
                shutil.copyfile(downloaded_path, staged)
                # Linking publishes the complete CSV atomically without overwriting.
                os.link(staged, destination)
            else:
                if not downloaded_path.is_dir():
                    raise ValueError("The run artifact download did not return a directory.")
                if destination.exists():
                    raise FileExistsError(destination)
                downloaded_path.rename(destination)
        print(f"  OK {destination}", flush=True)
        return "downloaded"
    except FileExistsError:
        print(f"  SKIP {destination} (already exists)", flush=True)
        return "skipped"
    except Exception as exc:
        print(f"  FAILED {name} ({run_id}): {exc}", file=sys.stderr)
        return "failed"


def download_experiment(client, experiment_name, output_dir, view_type, artifact_path,
                        download_mode="csv", *, client_factory, max_workers=8):
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    runs = list_runs(client, experiment.experiment_id, view_type)
    if not runs:
        raise ValueError(f"No runs found in experiment: {experiment_name}")

    names = [safe_name(run.data.tags.get("mlflow.runName") or run.info.run_id) for run in runs]
    counts = Counter(name.casefold() for name in names)
    used = set()
    worker_count = min(max_workers, len(runs))
    print(f"{experiment_name}: {len(runs)} runs, up to {worker_count} concurrent downloads", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve every output name before scheduling to keep names deterministic.
    jobs = []
    for run, name in zip(runs, names):
        run_id = run.info.run_id
        if counts[name.casefold()] > 1:
            name = f"{name}__{run_id}"
        # Also handle a run name that resembles another run's suffixed name.
        while name.casefold() in used:
            name += f"__{run_id}"
        used.add(name.casefold())
        destination = output_dir / (f"{name}.csv" if download_mode == "csv" else name)
        jobs.append((run_id, name, destination))

    results = Counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                download_run, client_factory, run_id, name, destination,
                artifact_path, download_mode,
            )
            for run_id, name, destination in jobs
        ]
        for future in as_completed(futures):
            results[future.result()] += 1

    print(
        f"{output_dir}: {results['downloaded']} downloaded, {results['skipped']} skipped, "
        f"{results['missing']} missing, {results['failed']} failed",
        flush=True,
    )
    return results["missing"] + results["failed"]


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
    def client_factory():
        return MlflowClient(tracking_uri=args.tracking_uri)

    client = client_factory()
    view_type = ViewType.ALL if args.include_deleted else ViewType.ACTIVE_ONLY
    problems = 0
    for experiment_name, folder in (
        (args.models_experiment, "models"),
        (args.ensembles_experiment, "ensembles"),
    ):
        try:
            problems += download_experiment(
                client, experiment_name, args.output_dir / folder, view_type,
                args.artifact_path, download_mode=args.download_mode,
                client_factory=client_factory, max_workers=args.max_workers,
            )
        except Exception as exc:
            problems += 1
            print(f"FAILED experiment {experiment_name}: {exc}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
