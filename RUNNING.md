# 运行说明

以下命令假设已按 [ENVIRONMENT.md](ENVIRONMENT.md) 配置好 Torch 和 Jittor。路径均通过专用变量传入，不依赖代码中的amax默认绝对路径。

```bash
export JIT_REPO=/path/to/JiT
export IMAGENET_ROOT=/path/to/imagenet-1k
export JIT_CHECKPOINT=/path/to/jib-b-16.pth
export JIT_OUTPUT_ROOT=/path/to/jit-outputs
export TORCH_PY=/path/to/jit-torch-env/bin/python
export JITTOR_PY=/path/to/jit-jittor-env/bin/python
```

## 数据目录

Torch 脚本会自动在 `--data_path` 后追加 `train`，因此需要：

```text
$IMAGENET_ROOT/
└── train/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

Jittor `train_benchmark.py` 需要直接传入 `"$IMAGENET_ROOT/train"`。两套实现都使用 ImageFolder 类目子目录。

## Jittor checkpoint 校验和单批推理

```bash
cd "$JIT_REPO/jit-jittor"

# 只构建模型并校验183个checkpoint tensor
CUDA_VISIBLE_DEVICES=0 "$JITTOR_PY" infer.py \
  --checkpoint "$JIT_CHECKPOINT" \
  --dry-run

# BF16、50-step Heun、分离conditional/unconditional CFG
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

输出包含逐图 PNG、`grid.png` 和 `metrics.json`。

## Jittor FID-50K

先在 Jittor 环境生成 50,000 张图，再在 Torch 环境使用与对照实现一致的 JiT-pinned `torch-fidelity` 评估：

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

`jit-jittor/run_fid50k.sh` 保留了amax历史路径，迁移到新机器时应优先使用上述显式参数命令。不要让多个进程同时写同一个 FID 图片目录。

## Jittor 单 A100 训练短 benchmark

`train_benchmark.py` 是短 benchmark，不保存 checkpoint。它的 `--monitor-script` 参数需要一个接受 `--pid`/`--output`/`--interval` 的资源监控脚本；在amax上可使用共享监控器：

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
  --warmup-epochs 5 \
  --monitor-script /mnt/nfs/home/xutianyi/JiT/benchmarks/jit_b16_a100/monitor_resources.py
```

## Jittor 双节点 / 双 A100 benchmark

该脚本的输出文件按 hostname 命名，所以已验证口径是“两台主机、每台一个 MPI rank”。不要在同一主机启动多个 rank，否则日志可能相互覆盖。

在可互相 SSH 的amax节点上，从主节点执行：

```bash
cd "$JIT_REPO/jit-jittor"

mpirun -np 2 --host amax1:1,amax4:1 --map-by ppr:1:node \
  -x PATH -x LD_LIBRARY_PATH -x CPLUS_INCLUDE_PATH -x nvcc_path -x JITTOR_HOME \
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
  --warmup-epochs 5 \
  --monitor-script /mnt/nfs/home/xutianyi/JiT/benchmarks/jit_b16_a100/monitor_resources.py
```

节点名、SSH、NCCL/RoCE 网卡选择和防火墙由集群环境决定；上述命令不使用任何 `sudo`。

## PyTorch 单卡训练

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

单卡下 `128 × accumulation 8 = effective batch 1024`。如果 `--resume` 目录不包含 `checkpoint-last.pth`，脚本从随机初始化开始。

## PyTorch 双节点 / 双 A100 训练

两台主机分别执行一条命令，`MASTER_ADDR`、`MASTER_PORT` 和共享路径必须一致。

节点0：

```bash
cd "$JIT_REPO/jit-torch"
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=amax1.rc4ml.org --master_port=29531 \
  main_jit_accum.py \
  --model JiT-B/16 --img_size 256 --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 --noise_scale 1.0 \
  --batch_size 128 --accum_iter 4 --blr 5e-5 \
  --epochs 200 --warmup_epochs 5 \
  --output_dir "$JIT_OUTPUT_ROOT/torch-two-node" \
  --resume "$JIT_OUTPUT_ROOT/torch-two-node" \
  --data_path "$IMAGENET_ROOT"
```

节点1使用相同参数，只将 `--node_rank=0` 改为 `--node_rank=1`。双卡下 `128 × 2 ranks × accumulation 4 = effective batch 1024`。

## PyTorch checkpoint 评估

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

`--resume` 接受的是目录，目录内需要存在 `checkpoint-last.pth`。

## 建议的运行前检查

```bash
test -d "$IMAGENET_ROOT/train"
test -f "$JIT_CHECKPOINT"
test -w "$JIT_OUTPUT_ROOT"
nvidia-smi
```

首次 Jittor 运行会编译并写入 `JITTOR_HOME`。正式性能统计应单独标记冷启动编译时间，不将它混入稳态吞吐。
