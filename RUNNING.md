# Running

Define repository, dataset, output, and Python paths before running:

```bash
export JIT_REPO=/path/to/JiT
export IMAGENET_ROOT=/path/to/imagenet-1k
export JIT_OUTPUT_ROOT=/path/to/jit-outputs
export TORCH_PY=/path/to/python
```

The dataset must contain `"$IMAGENET_ROOT/train"` in ImageFolder format.

## Single-GPU training

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

## Two-node training

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

## Checkpoint evaluation

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
