#!/usr/bin/env python3
"""Generate ImageNet samples from the official JiT-B/16 PyTorch checkpoint."""

import argparse
import gc
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import jittor as jt
import numpy as np
from PIL import Image

from denoiser import Denoiser


DEFAULT_CHECKPOINT = os.environ.get(
    "JIT_CHECKPOINT", "checkpoints/checkpoint-last.pth"
)
DEFAULT_OUTPUT = os.environ.get("JIT_OUTPUT_DIR", "outputs/jittor-inference")


class GpuMemoryMonitor:
    """Poll this process's GPU allocation as reported by nvidia-smi."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.pid = os.getpid()
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                for line in output.splitlines():
                    pid, memory = [item.strip() for item in line.split(",", 1)]
                    if int(pid) == self.pid:
                        self.peak_mib = max(self.peak_mib, int(memory))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def gpu_information():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        first = output.splitlines()[0]
        name, memory, driver = [item.strip() for item in first.split(",", 2)]
        return {
            "name": name,
            "total_memory_mib": int(memory),
            "driver": driver,
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {}


def load_ema(model, checkpoint_path, ema_key):
    started = time.perf_counter()
    checkpoint = jt.load(str(checkpoint_path))
    if ema_key not in checkpoint:
        raise KeyError(
            f"{ema_key!r} is absent; checkpoint keys: {list(checkpoint)}"
        )
    selected = checkpoint[ema_key]
    expected = model.state_dict()

    missing = sorted(set(expected) - set(selected))
    unexpected = sorted(set(selected) - set(expected))
    mismatched = [
        (name, tuple(expected[name].shape), tuple(selected[name].shape))
        for name in sorted(set(expected) & set(selected))
        if tuple(expected[name].shape) != tuple(selected[name].shape)
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint/model mismatch:\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}\n"
            f"shape_mismatches={mismatched}"
        )

    # Drop the two unused ~500 MB states before forcing CUDA synchronization.
    model.load_parameters(selected)
    parameter_count = sum(int(np.prod(value.shape)) for value in selected.values())
    del selected, checkpoint
    gc.collect()
    jt.sync_all()
    return {
        "ema_key": ema_key,
        "parameter_tensors": len(expected),
        "parameter_count": parameter_count,
        "load_seconds": time.perf_counter() - started,
    }


def save_images(samples, labels, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    array = samples.float32().numpy()
    array = np.clip((array + 1.0) * 127.5, 0, 255)
    array = np.rint(array).astype(np.uint8).transpose(0, 2, 3, 1)

    paths = []
    images = []
    for index, (pixels, label) in enumerate(zip(array, labels)):
        image = Image.fromarray(pixels, mode="RGB")
        path = output_dir / f"sample_{index:03d}_class_{label:04d}.png"
        image.save(path)
        paths.append(str(path))
        images.append(image)

    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * 256, rows * 256))
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * 256, (index // columns) * 256))
    grid_path = output_dir / "grid.png"
    grid.save(grid_path)
    return paths, str(grid_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=[207],
        help="Zero-based ImageNet-1K class IDs.",
    )
    parser.add_argument("--ema", choices=("model_ema1", "model_ema2"), default="model_ema1")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("heun", "euler"), default="heun")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Use BF16 model parameters to match the reference autocast run.",
    )
    parser.add_argument(
        "--cfg-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate conditional/unconditional branches in one batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the network and validate/load all checkpoint tensors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if any(label < 0 or label >= 1000 for label in args.labels):
        raise ValueError("Every --labels value must be in [0, 999]")

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    monitor = GpuMemoryMonitor()
    monitor.start()
    total_started = time.perf_counter()

    model = Denoiser(
        steps=args.steps,
        method=args.method,
        cfg_scale=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        cfg_batch=args.cfg_batch,
    )
    model.eval()
    load_metrics = load_ema(model, Path(args.checkpoint), args.ema)
    if args.precision == "bf16":
        model.bfloat16()
        jt.sync_all()
    print(
        f"Loaded {load_metrics['parameter_tensors']} tensors "
        f"({load_metrics['parameter_count']:,} parameters) from {args.ema} "
        f"in {load_metrics['load_seconds']:.2f}s",
        flush=True,
    )

    metadata = {
        "framework": f"jittor {jt.__version__}",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model": "JiT-B/16",
        "labels": args.labels,
        "seed": args.seed,
        "precision": args.precision,
        "sampler": {
            "method": args.method,
            "steps": args.steps,
            "cfg": args.cfg,
            "cfg_interval": [args.interval_min, args.interval_max],
            "cfg_batch": args.cfg_batch,
        },
        "gpu": gpu_information(),
        "load": load_metrics,
    }

    if not args.dry_run:
        labels = jt.array(np.asarray(args.labels, dtype=np.int32))
        inference_started = time.perf_counter()

        def progress(step, total):
            if step == 1 or step % 5 == 0 or step == total:
                print(f"Sampling step {step}/{total}", flush=True)

        with jt.no_grad():
            samples = model.generate(labels, progress_callback=progress)
        jt.sync_all()
        inference_seconds = time.perf_counter() - inference_started
        image_paths, grid_path = save_images(
            samples, args.labels, Path(args.output_dir)
        )
        metadata["inference"] = {
            "seconds": inference_seconds,
            "seconds_per_image": inference_seconds / len(args.labels),
            "images": image_paths,
            "grid": grid_path,
        }
        print(
            f"Generated {len(args.labels)} image(s) in "
            f"{inference_seconds:.2f}s; grid: {grid_path}",
            flush=True,
        )

    monitor.stop()
    metadata["gpu"]["peak_process_memory_mib"] = monitor.peak_mib
    metadata["total_seconds"] = time.perf_counter() - total_started
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Peak process GPU memory: {monitor.peak_mib} MiB; "
        f"metrics: {metrics_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
