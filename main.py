#!/usr/bin/env python3
"""Single public entry point for MAC-DTA checks, training, testing, and ablations."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MAIN_CONFIG = ROOT / "configs" / "mac_dta.json"
PATH_ARGUMENTS = {"csv_path", "teacher_pred_file", "train_pair_csv", "eval_pair_csv"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["arguments"] = dict(base.get("arguments", {}))
    merged["arguments"].update(overlay.get("arguments", {}))
    merged["flags"] = list(base.get("flags", [])) + list(overlay.get("flags", []))
    for key, value in overlay.items():
        if key not in {"arguments", "flags"}:
            merged[key] = value
    return merged


def load_experiment(name: str) -> dict[str, Any]:
    base = load_json(MAIN_CONFIG)
    if name == "mac_dta":
        return base
    overlay_path = ROOT / "configs" / "ablations" / f"{name}.json"
    if not overlay_path.exists():
        available = ", ".join(
            ["mac_dta"] + sorted(path.stem for path in overlay_path.parent.glob("*.json"))
        )
        raise FileNotFoundError(f"Unknown experiment '{name}'. Available: {available}")
    return merge_config(base, load_json(overlay_path))


def count_csv(path: Path, split_column: str) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = 0
        for row in reader:
            rows += 1
            counts[row.get(split_column, "<missing>")] += 1
    return rows, counts


def verify_repository(verbose: bool = True) -> list[str]:
    required = [
        "macdta/main_warm_kan.py",
        "macdta/model.py",
        "macdta/dataset.py",
        "macdta/mut_local.py",
        "macdta/cliff_loss.py",
        "macdta/cliff_eval.py",
        "data/davis/davis_processed_fixed.csv",
        "data/davis/mutation_info.csv",
        "data/davis/cliff_pairs.csv",
        "data/davis/augmented_pairs_near_wtmut.csv",
        "data/davis/unimol_compounds_davis_512.pt",
        "data/davis/teacher_predictions.npy",
        "data/davis/Vocab/smiles_vocab.pkl",
        "data/davis/Vocab/protein_vocab.pkl",
    ]
    errors = [f"missing: {item}" for item in required if not (ROOT / item).is_file()]

    graph_dir = ROOT / "data" / "davis" / "graphs_davis"
    graph_count = len(list(graph_dir.glob("*.pt"))) if graph_dir.is_dir() else 0
    if graph_count != 374:
        errors.append(f"expected 374 protein graph files, found {graph_count}")

    dataset_path = ROOT / "data" / "davis" / "davis_processed_fixed.csv"
    if dataset_path.exists():
        rows, splits = count_csv(dataset_path, "split")
        expected = Counter({"train": 24077, "val": 2985, "test": 2994})
        if rows != 30056 or splits != expected:
            errors.append(f"unexpected Davis rows/splits: rows={rows}, splits={dict(splits)}")

    cliff_path = ROOT / "data" / "davis" / "cliff_pairs.csv"
    if cliff_path.exists():
        rows, splits = count_csv(cliff_path, "pair_split")
        expected = Counter({"train": 384, "val": 65, "test": 56})
        if rows != 505 or splits != expected:
            errors.append(f"unexpected cliff-pair counts: rows={rows}, splits={dict(splits)}")

    augmented_path = ROOT / "data" / "davis" / "augmented_pairs_near_wtmut.csv"
    if augmented_path.exists():
        rows, splits = count_csv(augmented_path, "pair_split")
        if rows != 838 or splits.get("train") != 648:
            errors.append(
                f"unexpected augmented-pair counts: rows={rows}, splits={dict(splits)}"
            )

    if verbose:
        status = "OK" if not errors else "FAILED"
        print(f"[repository check] {status}")
        print("  Davis rows: 30,056 (train 24,077 / val 2,985 / test 2,994)")
        print("  Evaluation cliff pairs: 505 (train 384 / val 65 / test 56)")
        print("  Pairwise training relations: 648")
        print(f"  Protein graphs: {graph_count}")
        for error in errors:
            print(f"  ERROR: {error}")
    return errors


def print_environment() -> None:
    package_names = {
        "torch": "torch",
        "torch-geometric": "torch-geometric",
        "torch-scatter": "torch-scatter",
        "rdkit": "rdkit",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "scipy": "scipy",
        "tqdm": "tqdm",
        "networkx": "networkx",
    }
    print(f"Python: {sys.version.split()[0]} (repository target: 3.10)")
    for label, distribution in package_names.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        print(f"{label}: {version}")


def select_device(value: str) -> int:
    if value == "cpu":
        return -1
    if value != "auto":
        return int(value)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed; run pip install -r requirements.txt") from exc
    return 0 if torch.cuda.is_available() else -1


def resolve_argument(key: str, value: Any) -> Any:
    if key in PATH_ARGUMENTS and value is not None:
        path = Path(str(value))
        return str(path if path.is_absolute() else (ROOT / path).resolve())
    return value


def build_command(
    config: dict[str, Any],
    experiment: str,
    seed: int,
    mode: str,
    device: int,
    checkpoint: Path | None,
    output_dir: Path | None,
    epochs: int | None,
    batch_size: int | None,
    num_workers: int | None,
    max_train_batches: int | None,
) -> list[str]:
    arguments = dict(config.get("arguments", {}))
    if epochs is not None:
        arguments["epochs"] = epochs
    if batch_size is not None:
        arguments["batch_size"] = batch_size
    if num_workers is not None:
        arguments["num_workers"] = num_workers
    if max_train_batches is not None:
        arguments["max_train_batches"] = max_train_batches

    command = [sys.executable, "-u", str(ROOT / "macdta" / "main_warm_kan.py")]
    command.extend(["--seed", str(seed), "--cuda_device", str(device)])
    command.extend(["--exp_name", f"{experiment}_s{seed}"])

    for key, raw_value in arguments.items():
        value = resolve_argument(key, raw_value)
        if value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            command.append(flag)
        else:
            command.extend([flag, str(value)])

    for flag in config.get("flags", []):
        command.append(flag if str(flag).startswith("--") else f"--{flag}")

    if mode == "test":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --mode test")
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        command.extend(
            [
                "--eval_only",
                "--resume",
                str(checkpoint),
                "--output_dir",
                str(checkpoint.parent),
                "--ckpt_base_name",
                checkpoint.stem,
            ]
        )
    else:
        run_output = output_dir or (ROOT / "outputs" / experiment / f"seed{seed}")
        run_output = run_output.expanduser().resolve()
        run_output.mkdir(parents=True, exist_ok=True)
        command.extend(["--output_dir", str(run_output)])

    return command


def run_command(command: list[str], dry_run: bool) -> int:
    print(f"[command] {shlex.join(command)}")
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command MAC-DTA repository check, training, test, and ablation runner."
    )
    parser.add_argument(
        "--mode",
        choices=["check", "train", "test", "ablation"],
        default="train",
    )
    parser.add_argument("--experiment", default="mac_dta")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a visible device index")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-environment", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify_repository(verbose=True)
    if args.show_environment:
        print_environment()
    if args.mode == "check":
        return 1 if errors else 0
    if errors:
        print("Training/test stopped because the repository check failed.", file=sys.stderr)
        return 1

    config = load_experiment(args.experiment)
    device = select_device(args.device)
    mode = "train" if args.mode == "ablation" else args.mode
    command = build_command(
        config=config,
        experiment=args.experiment,
        seed=args.seed,
        mode=mode,
        device=device,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
    )
    return run_command(command, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
