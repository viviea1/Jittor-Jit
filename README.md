# JiT PyTorch / Jittor 对照仓库

本仓库在 2026-08-08 从 amax 共享目录创建，保留两套互相独立的 JiT-B/16 实现：

| 目录 | 来源 | 用途 |
|---|---|---|
| `jit-torch/` | `/mnt/nfs/home/xutianyi/JiT/jit-torch` | PyTorch 训练、续训、生成与 FID/IS 评估 |
| `jit-jittor/` | `/mnt/nfs/home/xutianyi/JiT/jit-jittor` | Jittor 推理、FID-50K 生成和受控训练 benchmark |

下载时排除了远程嵌套 `.git`、`__pycache__` 和 `.pyc`；当前目录是唯一的 Git 仓库根目录。

## 文档

- [环境说明](ENVIRONMENT.md)：amax 已验证的软硬件版本、Conda 安装和 Jittor CUDA/cuDNN 环境变量。
- [运行说明](RUNNING.md)：数据布局、Torch 训练/续训/评估、Jittor 推理/FID 和双 A100 benchmark。
- [PyTorch 上游说明](jit-torch/README.md)
- [Jittor 实现说明](jit-jittor/README.md)

## 能力边界

- `jit-torch/main_jit.py` 提供完整训练、checkpoint 和在线评估。
- `jit-torch/main_jit_accum.py` 在此基础上增加梯度累积，用于双 A100 上保持 effective batch。
- `jit-torch/main_jit_single_resume.py` 支持限制每 epoch micro-step，用于单卡续训和短测。
- `jit-jittor/train_benchmark.py` 是真实 ImageNet 训练短 benchmark：执行 BF16 前向、FP32 master AdamW、梯度同步和双 EMA，但不包含完整 epoch 循环、checkpoint 写入或训练后 FID 联动。
- amax 上正在运行的完整 Jittor epoch 脚本位于共享目录的另一个 `JiT/tools/` 目录，不在本次指定的两个源目录中，因此本仓库当前不应声称已包含可恢复的 Jittor 完整长训流程。

## 数据和大文件

ImageNet、checkpoint、生成图像和实验输出不进入 Git。建议在仓库外管理，并通过命令行参数传入。`jit-torch/fid_stats/` 中的两个预计算统计文件已从远程源保留。

## 许可边界

`jit-torch/` 包含其上游 `LICENSE`。远程 `jit-jittor/` 源目录未提供独立许可文件；在对外发布 Jittor 部分前，需要确认其派生代码的授权和归属。
