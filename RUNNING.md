# Running

Define repository, dataset, output, and Python paths before running:

```bash
export JIT_REPO=/path/to/JiT
export IMAGENET_ROOT=/path/to/imagenet-1k
export JIT_OUTPUT_ROOT=/path/to/jit-outputs
export JIT_CHECKPOINT=/path/to/checkpoint-last.pth
export TORCH_PY=/path/to/python
export JITTOR_PY=/path/to/jittor-python
```

The dataset must contain `"$IMAGENET_ROOT/train"` in ImageFolder format.

## PyTorch

### Single-GPU training

```bash
cd "$JIT_REPO/jit-torch"

CUDA_VISIBLE_DEVICES=0 "$TORCH_PY" main_jit_accum.py \
  --model JiT-B/16 \
  --img_size 256 \
  --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 \
  --noise_scale 1.0 \
  --batch_size 128 \
  --accum_iter 8 \
  --blr 5e-5 \
  --epochs 200 \
  --warmup_epochs 5 \
  --output_dir "$JIT_OUTPUT_ROOT/torch-single" \
  --resume "$JIT_OUTPUT_ROOT/torch-single" \
  --data_path "$IMAGENET_ROOT"
```

### Two-node training

Run the following command on both nodes with the same shared paths and rendezvous address. Set `NODE_RANK=0` on the first node and `NODE_RANK=1` on the second.

```bash
export NODE_RANK=0
export MASTER_ADDR=first-node.example
export MASTER_PORT=29531

cd "$JIT_REPO/jit-torch"

torchrun --nnodes=2 --nproc_per_node=1 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  main_jit_accum.py \
  --model JiT-B/16 \
  --img_size 256 \
  --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 \
  --noise_scale 1.0 \
  --batch_size 128 \
  --accum_iter 4 \
  --blr 5e-5 \
  --epochs 200 \
  --warmup_epochs 5 \
  --output_dir "$JIT_OUTPUT_ROOT/torch-two-node" \
  --resume "$JIT_OUTPUT_ROOT/torch-two-node" \
  --data_path "$IMAGENET_ROOT"
```

### Checkpoint evaluation

`--resume` must point to a directory containing `checkpoint-last.pth`.

```bash
cd "$JIT_REPO/jit-torch"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 main_jit.py \
  --model JiT-B/16 \
  --img_size 256 \
  --noise_scale 1.0 \
  --gen_bsz 64 \
  --num_images 50000 \
  --cfg 3.0 \
  --interval_min 0.1 \
  --interval_max 1.0 \
  --output_dir "$JIT_OUTPUT_ROOT/torch-eval" \
  --resume /path/to/checkpoint-directory \
  --data_path "$IMAGENET_ROOT" \
  --evaluate_gen
```

## Jittor

The included training entry point is a bounded benchmark. It exercises the
forward pass, backward pass, FP32 master-weight AdamW update, gradient
accumulation, distributed gradient averaging, and dual EMA, but it does not
save a training checkpoint or implement a complete epoch schedule.

### Checkpoint validation

```bash
cd "$JIT_REPO/jit-jittor"

CUDA_VISIBLE_DEVICES=0 "$JITTOR_PY" infer.py \
  --checkpoint "$JIT_CHECKPOINT" \
  --dry-run
```

### Image generation

```bash
cd "$JIT_REPO/jit-jittor"

CUDA_VISIBLE_DEVICES=0 "$JITTOR_PY" infer.py \
  --checkpoint "$JIT_CHECKPOINT" \
  --output-dir "$JIT_OUTPUT_ROOT/jittor-infer" \
  --labels 0 207 999 \
  --ema model_ema1 \
  --steps 50 \
  --method heun \
  --cfg 3.0 \
  --interval-min 0.1 \
  --interval-max 1.0 \
  --precision bf16 \
  --no-cfg-batch
```

### FID-50K generation and evaluation

```bash
cd "$JIT_REPO/jit-jittor"

CUDA_VISIBLE_DEVICES=0 "$JITTOR_PY" fid50k.py \
  --checkpoint "$JIT_CHECKPOINT" \
  --output-dir "$JIT_OUTPUT_ROOT/jittor-fid50k" \
  --num-images 50000 \
  --batch-size 64 \
  --steps 50 \
  --cfg 3.0 \
  --interval-min 0.1 \
  --interval-max 1.0 \
  --seed 0 \
  --ema model_ema1 \
  --precision bf16 \
  --resume

CUDA_VISIBLE_DEVICES=0 "$TORCH_PY" evaluate_fid.py \
  --input-dir "$JIT_OUTPUT_ROOT/jittor-fid50k/images" \
  --stats "$JIT_REPO/jit-torch/fid_stats/jit_in256_stats.npz" \
  --output "$JIT_OUTPUT_ROOT/jittor-fid50k/fid_results.json"
```

The same pipeline can be launched through `run_fid50k.sh` by defining
`JIT_OUTPUT_DIR`, `JIT_CHECKPOINT`, `JITTOR_PY`, and `TORCH_PY`.

### Single-GPU training benchmark

```bash
cd "$JIT_REPO/jit-jittor"

CUDA_VISIBLE_DEVICES=0 "$JITTOR_PY" train_benchmark.py \
  --data-path "$IMAGENET_ROOT/train" \
  --checkpoint "$JIT_CHECKPOINT" \
  --output-dir "$JIT_OUTPUT_ROOT/jittor-single-benchmark" \
  --micro-batch 64 \
  --accumulation-steps 16 \
  --rounds 3 \
  --steps-per-round 192 \
  --num-workers 8 \
  --lr 2e-4 \
  --warmup-epochs 5
```

### Two-node training benchmark

The repository and data paths must be available under the same paths on both
nodes. Set `MPI_HOSTS` to two reachable hosts with one process per host.

```bash
export MPI_HOSTS=first-node.example:1,second-node.example:1

cd "$JIT_REPO/jit-jittor"

mpirun -np 2 --host "$MPI_HOSTS" --map-by ppr:1:node \
  "$JITTOR_PY" train_benchmark.py \
  --data-path "$IMAGENET_ROOT/train" \
  --checkpoint "$JIT_CHECKPOINT" \
  --output-dir "$JIT_OUTPUT_ROOT/jittor-two-node-benchmark" \
  --micro-batch 64 \
  --accumulation-steps 8 \
  --rounds 3 \
  --steps-per-round 192 \
  --num-workers 12 \
  --lr 2e-4 \
  --warmup-epochs 5
```
