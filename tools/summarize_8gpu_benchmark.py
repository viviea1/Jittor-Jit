#!/usr/bin/env python3
"""Summarize archived AutoDL Jittor MPI benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FULL_IMAGENET_IMAGES = 1_281_167
EPOCHS = 200


def number(value: str) -> float:
    return float(value.strip().split()[0])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize_run(run_dir: Path) -> dict:
    config_path = next(run_dir.glob("config_*_rank0.json"))
    config = json.loads(config_path.read_text())
    event_paths = sorted(run_dir.glob("events_*_rank*.jsonl"))
    rank_events = {path.stem.rsplit("rank", 1)[-1]: read_jsonl(path) for path in event_paths}
    rank0 = rank_events["0"]
    rounds = [event for event in rank0 if event["event"] == "round_end"]
    steady = [event["samples_per_second"] for event in rounds if event["round"] > 0]
    run_ends = [
        event
        for events in rank_events.values()
        for event in events
        if event["event"] == "run_end"
    ]

    gpu_stats: dict[str, dict[str, list[float]]] = {}
    with (run_dir / "gpu_global.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = row[" index"].strip()
            values = gpu_stats.setdefault(index, {"memory": [], "util": [], "power": [], "temp": []})
            values["memory"].append(number(row[" memory.used [MiB]"]))
            values["util"].append(number(row[" utilization.gpu [%]"]))
            values["power"].append(number(row[" power.draw [W]"]))
            values["temp"].append(number(row[" temperature.gpu"]))

    per_gpu_peak_memory = {
        index: max(values["memory"]) for index, values in sorted(gpu_stats.items())
    }
    effective_batch = int(config["effective_batch"])
    effective_samples_per_epoch = (FULL_IMAGENET_IMAGES // effective_batch) * effective_batch
    total_samples = effective_samples_per_epoch * EPOCHS
    slow = min(steady)
    fast = max(steady)
    mean = statistics.mean(steady)

    return {
        "run": run_dir.name,
        "exit_code": int((run_dir / "exit_code.txt").read_text().strip()),
        "world_size": config["world_size"],
        "precision": config["precision"]["forward_parameters_and_activations"],
        "micro_batch_per_rank": config["micro_batch_per_rank"],
        "accumulation_steps": config["accumulation_steps"],
        "effective_batch": effective_batch,
        "trainable_parameters": config["trainable_parameters"],
        "dataset_source": config["data_source"],
        "benchmark_dataset_images": config["dataset_size"],
        "benchmark_dataset_scope": "first official ImageNet class (n01440764), repeated",
        "completed_ranks": len(run_ends),
        "micro_steps_per_rank": run_ends[0]["micro_steps"],
        "optimizer_steps_per_rank": run_ends[0]["optimizer_steps"],
        "round_throughput_img_s": [event["samples_per_second"] for event in rounds],
        "steady_throughput_img_s": {
            "min": slow,
            "mean": mean,
            "max": fast,
        },
        "observed_peak_memory_mib": {
            "per_gpu": per_gpu_peak_memory,
            "max": max(per_gpu_peak_memory.values()),
        },
        "full_imagenet_estimate": {
            "dataset_images": FULL_IMAGENET_IMAGES,
            "effective_samples_per_epoch": effective_samples_per_epoch,
            "epochs": EPOCHS,
            "total_effective_samples": total_samples,
            "pure_training_days_at_steady_min": total_samples / slow / 86400,
            "pure_training_days_at_steady_mean": total_samples / mean / 86400,
            "pure_training_days_at_steady_max": total_samples / fast / 86400,
        },
        "evidence": {
            "config": str(config_path),
            "events_rank0": str(next(run_dir.glob("events_*_rank0.jsonl"))),
            "gpu_samples": str(run_dir / "gpu_global.csv"),
            "mpi_log": str(run_dir / "mpirun.log"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        summarize_run(path)
        for path in sorted(args.evidence_dir.glob("run_fp32_*"))
        if path.is_dir()
    ]
    result = {
        "schema_version": 1,
        "scope": "AutoDL 8x A800 Jittor FP32 real-image training throughput smoke",
        "runs": runs,
        "interpretation": {
            "recommended_run": "run_fp32_m64_a2_s32_r4",
            "measured_boundary": (
                "Real ImageNet JPEG decode, JiT-B/16 forward/backward, NCCL gradient mean, "
                "FP32 AdamW master update, and two EMA updates all executed."
            ),
            "not_measured": (
                "Full 1000-class public-tar random I/O, checkpoint writes, 200-epoch "
                "stability, BF16 backward, and post-training FID/IS."
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
