# JiT-B/16 Jittor inference

This is a minimal inference-only Jittor port. It recreates the JiT-B/16
network, loads the official PyTorch `.pth` checkpoint directly through
Jittor, selects `model_ema1`, and runs the paper's default 50-step Heun
sampler with CFG 3.0.

The implementation does not depend on PyTorch and does not yet contain
training or FID code.

## Environment

- Python 3.10.20
- Jittor 1.3.11.0
- CUDA 12.2 / cuDNN 8
- NumPy 1.22.4
- Pillow 9.4.0

The existing environment is:

```bash
conda activate /home/xutianyi/miniconda3/envs/jit-jittor
```

The shared repository path is:

```text
/mnt/nfs/home/xutianyi/JiT/jit-jittor
```

Each compute node still needs its own Jittor conda environment and local
JIT cache. Before BF16 inference on a new node, apply the included
Jittor/cuDNN compatibility patch once:

```bash
/home/xutianyi/miniconda3/envs/jit-jittor/bin/python \
  patches/apply_jittor_bf16_cudnn_patch.py
```

Use `--precision bf16 --no-cfg-batch` to match the reference Torch
evaluation's BF16 model forward and separate conditional/unconditional
CFG calls.

The local Jittor 1.3.11 cuDNN forward-convolution wrapper is patched to
use FP32 accumulation for BF16 inputs. This is required by cuDNN 8 on the
A100 and matches Torch's BF16 autocast convolution behavior.

## Validate the checkpoint

```bash
cd /mnt/nfs/home/xutianyi/JiT/jit-jittor
python infer.py --dry-run
```

This checks all parameter names and shapes before loading. The checkpoint
contains 183 model tensors.

## Generate an image

```bash
cd /mnt/nfs/home/xutianyi/JiT/jit-jittor
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint /mnt/nfs/home/xutianyi/JiT/checkpoints/checkpoint-last.pth \
  --output-dir /mnt/nfs/home/xutianyi/JiT/outputs/jittor-inference \
  --labels 207 \
  --steps 50 \
  --method heun \
  --cfg 3.0
```

`--labels` uses zero-based ImageNet-1K class IDs. Multiple labels form a
batch, for example `--labels 207 360 387 971`.

By default, conditional and unconditional CFG branches are concatenated
into one Jittor batch. Use `--no-cfg-batch` to reproduce the two separate
network calls in the original code. The generated PNG files, `grid.png`,
and a timing/GPU-memory report in `metrics.json` are written to the output
directory.

## Validation on amax4

The port was checked against the PyTorch network with the same EMA1
weights and input. A full 256x256 forward pass had mean absolute error
`1.36e-4` and maximum absolute error `1.06e-3`.

On the NVIDIA A100-PCIE-40GB, one image with 50-step Heun sampling used
2272 MiB peak process GPU memory. The first full run took 17.60 seconds
including Jittor compilation; a subsequent cached run took 2.33 seconds.

## FID-50K

`fid50k.py` generates 50 images for each of the 1000 ImageNet classes and
supports resuming a partially written image folder. `evaluate_fid.py` is
then run in the Torch environment solely for the JiT-pinned
`torch-fidelity` evaluator. The full monitored pipeline is:

```bash
bash run_fid50k.sh
```

Do not run multiple workers against the same image/output directory.
Give every GPU or rank a disjoint output shard and seed; merge the
numbered images only after every shard completes.
