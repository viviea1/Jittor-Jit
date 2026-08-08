#!/usr/bin/env python3
"""Generate a class-balanced ImageNet sample folder with Jittor."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import jittor as jt
import numpy as np
from PIL import Image

from denoiser import Denoiser
from infer import DEFAULT_CHECKPOINT, load_ema


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema", choices=("model_ema1", "model_ema2"), default="model_ema1")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def contiguous_image_count(images_dir):
    count = 0
    while (images_dir / f"{count:05d}.png").is_file():
        count += 1
    return count


def save_batch(samples, start_index, images_dir):
    array = samples.float32().numpy()
    array = np.clip((array + 1.0) * 127.5, 0, 255)
    array = np.rint(array).astype(np.uint8).transpose(0, 2, 3, 1)
    for offset, pixels in enumerate(array):
        Image.fromarray(pixels, mode="RGB").save(
            images_dir / f"{start_index + offset:05d}.png"
        )


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.num_images < 1 or args.num_images % 1000:
        raise ValueError("--num-images must be a positive multiple of 1000")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    existing = contiguous_image_count(images_dir)
    if existing and not args.resume:
        raise RuntimeError(
            f"{images_dir} already contains {existing} contiguous images; "
            "pass --resume or choose a new output directory"
        )
    if existing > args.num_images:
        raise RuntimeError("Existing image count exceeds --num-images")

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    model = Denoiser(
        steps=args.steps,
        method="heun",
        cfg_scale=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        noise_scale=1.0,
        cfg_batch=False,
    ).eval()
    load_info = load_ema(model, Path(args.checkpoint), args.ema)
    if args.precision == "bf16":
        model.bfloat16()
        jt.sync_all()

    parameter_dtypes = sorted({str(value.dtype) for value in model.state_dict().values()})
    if args.precision == "bf16" and parameter_dtypes != ["bfloat16"]:
        raise RuntimeError(f"Expected all BF16 parameters, got {parameter_dtypes}")

    labels_all = np.arange(1000, dtype=np.int32).repeat(args.num_images // 1000)
    configuration = {
        "framework": f"jittor {jt.__version__}",
        "pid": os.getpid(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "ema": args.ema,
        "model": "JiT-B/16",
        "image_size": 256,
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "method": "heun",
        "cfg": args.cfg,
        "cfg_interval": [args.interval_min, args.interval_max],
        "noise_scale": 1.0,
        "seed": args.seed,
        "precision": args.precision,
        "cfg_batch": False,
        "parameter_dtypes": parameter_dtypes,
        "load": load_info,
    }
    write_json(output_dir / "generation_config.json", configuration)
    print(json.dumps(configuration, ensure_ascii=False), flush=True)
    print(f"Resuming from {existing}/{args.num_images} images", flush=True)

    started = time.perf_counter()
    total_batches = math.ceil(args.num_images / args.batch_size)
    with jt.no_grad():
        for batch_index, start in enumerate(
            range(0, args.num_images, args.batch_size)
        ):
            end = min(start + args.batch_size, args.num_images)
            batch_labels = labels_all[start:end]

            # Generate noise even for completed batches. This advances Jittor's
            # RNG stream so a resumed job follows the uninterrupted sequence.
            noise = jt.randn((end - start, 3, 256, 256))
            noise.sync()
            batch_complete = all(
                (images_dir / f"{index:05d}.png").is_file()
                for index in range(start, end)
            )
            if batch_complete:
                continue

            labels = jt.array(batch_labels)
            batch_started = time.perf_counter()
            samples = model.generate(labels, initial_noise=noise)
            save_batch(samples, start, images_dir)
            batch_seconds = time.perf_counter() - batch_started
            elapsed = time.perf_counter() - started
            completed = end
            rate = max(completed - existing, 0) / max(elapsed, 1e-9)
            eta = (args.num_images - completed) / max(rate, 1e-9)
            progress = {
                "completed": completed,
                "total": args.num_images,
                "batch": batch_index + 1,
                "total_batches": total_batches,
                "batch_seconds": batch_seconds,
                "elapsed_seconds_this_run": elapsed,
                "images_per_second_this_run": rate,
                "eta_seconds": eta,
            }
            write_json(output_dir / "generation_progress.json", progress)
            print(
                f"Generated {completed}/{args.num_images} "
                f"(batch {batch_index + 1}/{total_batches}, "
                f"{batch_seconds:.2f}s, {rate:.2f} img/s, "
                f"ETA {eta / 60:.1f} min)",
                flush=True,
            )

    final_count = contiguous_image_count(images_dir)
    if final_count != args.num_images:
        raise RuntimeError(
            f"Expected {args.num_images} contiguous images, found {final_count}"
        )
    result = {
        "completed": final_count,
        "generation_seconds_this_run": time.perf_counter() - started,
        "images_dir": str(images_dir),
    }
    write_json(output_dir / "generation_complete.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
