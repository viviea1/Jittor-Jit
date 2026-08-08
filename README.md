# Jittor-JiT

Jittor implementation and validation harness for JiT-B/16 pixel-space
diffusion. The repository covers checkpoint-compatible inference, FID-50K
generation, real-image training benchmarks, multi-GPU MPI/NCCL smoke tests,
and controlled Torch/Jittor numerical comparisons. It is no longer an
inference-only port.

The Torch reference code under `torch_reference/` comes from
[LTH14/JiT](https://github.com/LTH14/JiT). The Jittor port preserves the model
shape and checkpoint naming needed to load the reference EMA weights.

## Status and evidence boundaries

The results below were completed on the stated configurations. Short runs are
reported as smoke tests or benchmarks, not as evidence of long-run convergence.

| Validation | Result | Boundary |
| --- | --- | --- |
| Torch pretrained FID-50K | FID 3.6580, IS 269.4181 | ImageNet-256, 50K, BF16, Heun-50, CFG 3.0 |
| Jittor pretrained FID-50K | FID 3.680811, IS 269.0536 +/- 3.7150 | Jittor generation; the pinned Torch-Fidelity evaluator computes FID/IS |
| Fair FP32 training comparison | 10 steps completed in both frameworks; first-step loss difference 0.00007915; maximum 10-step loss difference 0.00050414 | Fixed 8-image batch, identical initialization and random inputs; numerical-path test, not a throughput claim |
| Jittor 8 x A800 FP32 smoke | 8 MPI ranks completed; longer smoke averaged 486.761 image/s | Repeated images from one ImageNet class; not 1000-class random I/O or 200-epoch stability |
| Torch 200 epoch training | FID 4.7355, IS 221.5594 | Torch-only result |
| Jittor 200 epoch training | Not completed | No Jittor 200-epoch FID is claimed |
| CUDA 12.8 BF16 training | Blocked in Jittor 1.3.11 backward code generation | `atomicAdd(jittor::bfloat16*, ...)` has no matching CUDA overload; FP32 training works |

Selected compact comparison tables and the loss curve are in
`docs/experiments/minimal_closure_20260808/`. Raw machine logs and internal
experiment manifests are intentionally excluded from the first release.

## Environment

The validated Jittor environment used:

- Python 3.10.20
- Jittor 1.3.11.0
- NumPy 1.22.4
- Pillow 9.4.0
- CUDA 12.2 with cuDNN 8 for the validated BF16 A100 path

A minimal environment can be created with:

```bash
conda create -n jittor-jit python=3.10.20 -y
conda activate jittor-jit
pip install -r requirements.txt
```

Jittor compiles kernels on first use. Give every machine a writable local cache
and confirm that CUDA, cuDNN headers, and `nvidia-smi` are visible. The scripts
under `patches/` are narrowly scoped, idempotent compatibility patches for the
validated Jittor 1.3.11 stack; inspect them before applying to a different
Jittor release.

The Torch reference environment is described by
`torch_reference/environment.yaml`.

## Data preparation

For an extracted ImageNet training set:

```bash
export IMAGENET_PATH=/path/to/imagenet-1k/train
```

`train_benchmark.py` also supports the official nested ImageNet training tar.
Build an offset index once, then pass both files to the benchmark:

```bash
python tools/nested_tar_dataset.py \
  --tar /path/to/ILSVRC2012_img_train.tar \
  --output data/imagenet_train_index.npz

python train_benchmark.py \
  --data-tar /path/to/ILSVRC2012_img_train.tar \
  --data-index data/imagenet_train_index.npz \
  --output-dir outputs/tar-smoke \
  --precision fp32 \
  --micro-batch 1 \
  --accumulation-steps 1 \
  --rounds 1 \
  --steps-per-round 1
```

ImageNet is not distributed with this repository. Follow its license and access
terms.

## Checkpoints and inference

Checkpoints are intentionally excluded from Git. Supply one explicitly or set
`JIT_CHECKPOINT`:

```bash
export JIT_CHECKPOINT=/path/to/jib-b-16.pth

python infer.py \
  --checkpoint "$JIT_CHECKPOINT" \
  --output-dir outputs/inference \
  --labels 207 \
  --steps 50 \
  --method heun \
  --cfg 3.0 \
  --precision bf16 \
  --no-cfg-batch
```

Use `python infer.py --checkpoint "$JIT_CHECKPOINT" --dry-run` to build the
network and validate all checkpoint tensor names and shapes. The validated
JiT-B/16 EMA checkpoint contains 183 tensors.

## Training benchmark and long-training boundary

A single-process real-image smoke can be run with:

```bash
python train_benchmark.py \
  --data-path "$IMAGENET_PATH" \
  --output-dir outputs/train-smoke \
  --precision fp32 \
  --micro-batch 1 \
  --accumulation-steps 1 \
  --rounds 1 \
  --steps-per-round 1
```

For MPI, launch one process per GPU using the MPI installation visible to
Jittor, for example:

```bash
mpirun -np 8 python train_benchmark.py \
  --data-path "$IMAGENET_PATH" \
  --output-dir outputs/8gpu-smoke \
  --precision fp32 \
  --micro-batch 64 \
  --accumulation-steps 2 \
  --rounds 1 \
  --steps-per-round 8
```

The benchmark exercises JPEG decode, JiT-B/16 forward/backward, distributed
gradient averaging, FP32 AdamW master weights, and two EMA copies. It does not
provide checkpoint/resume orchestration for a production 200-epoch run. The
recorded 8 x A800 tests are throughput and stability smokes only. A formal long
run additionally needs complete 1000-class sampling, periodic atomic
checkpoints, restart validation, long-horizon loss monitoring, and final
FID-50K evaluation.

## FID-50K

`fid50k.py` writes resumable numbered samples. `evaluate_fid.py` evaluates the
folder with the pinned Torch-Fidelity path. The monitored wrapper is configured
entirely through environment variables:

```bash
export JITTOR_PY=/path/to/jittor-env/bin/python
export TORCH_PY=/path/to/torch-env/bin/python
export JIT_CHECKPOINT=/path/to/jib-b-16.pth
export FID_STATS=/path/to/jit_in256_stats.npz
export OUT_DIR="$PWD/outputs/jittor-b16-256-fid50k"
bash run_fid50k.sh
```

The two upstream reference-stat files are excluded from the first Git commit.
Obtain them from the `fid_stats/` directory of the upstream
[LTH14/JiT](https://github.com/LTH14/JiT) repository and verify:

```text
412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c  jit_in256_stats.npz
cdd15b54f9d1f26a881fcdc920a3410fa7a934047d6c4f5b8395b87182a9ab42  jit_in512_stats.npz
```

Do not run multiple workers against the same image directory. Give each rank a
disjoint output shard and seed, then merge only after all ranks finish.

## Repository contents

- `model_jit.py`, `denoiser.py`: Jittor JiT-B/16 and sampling code
- `infer.py`, `fid50k.py`, `evaluate_fid.py`: inference and FID pipeline
- `train_benchmark.py`, `mpi_probe.py`: training and distributed smoke tests
- `tools/`: dataset indexing, resource monitoring, and result summarization
- `patches/`: Jittor 1.3.11 compatibility patches
- `torch_reference/`: upstream Torch reference code and environment
- `docs/`: experiment summaries and selected compact evidence

## License

The upstream Torch implementation is MIT licensed. The root `LICENSE` reuses
the verified upstream MIT text and copyright notice. Keep the upstream notice
when redistributing derived code.
