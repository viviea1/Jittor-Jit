#!/usr/bin/env python3
"""Evaluate a generated image folder with JiT's pinned torch-fidelity."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_fidelity


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    metrics = torch_fidelity.calculate_metrics(
        input1=args.input_dir,
        input2=None,
        fid_statistics_file=args.stats,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    result = {
        key: float(value) if hasattr(value, "__float__") else value
        for key, value in metrics.items()
    }
    result.update(
        {
            "evaluator": "JiT-pinned torch-fidelity",
            "torch_version": torch.__version__,
            "input_dir": str(Path(args.input_dir).resolve()),
            "stats": str(Path(args.stats).resolve()),
            "evaluation_seconds": time.perf_counter() - started,
        }
    )
    output = Path(args.output)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

