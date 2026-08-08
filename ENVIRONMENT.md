# 环境说明

## amax8 已验证环境

以下版本由 `amax8.rc4ml.org` 实际查询，不是仅从依赖文件推断：

| 项目 | Jittor 环境 | PyTorch 环境 |
|---|---|---|
| GPU | NVIDIA A100-PCIE-40GB | NVIDIA A100-PCIE-40GB |
| NVIDIA driver | 580.126.09 | 580.126.09 |
| Python | 3.10.20 | 3.10.20 |
| 框架 | Jittor 1.3.11.0 | PyTorch 2.5.1 |
| CUDA 工具链 | 12.2.140 | PyTorch CUDA 12.4 |
| cuDNN | 8（Jittor 共享工具链） | 9.1.0 |
| NumPy | 1.22.4 | 1.22.4 |
| Pillow / torchvision | Pillow 9.4.0 | torchvision 0.20.1 |
| 分布式组件 | Open MPI 5.0.6 | `torch.distributed` / NCCL |

amax8 上已验证的 Python 路径：

```text
/home/xutianyi/miniconda3/envs/jit-jittor/bin/python
/home/xutianyi/miniconda3/envs/jit-torch/bin/python
```

## PyTorch 环境安装

Linux/NVIDIA 机器上：

```bash
cd /path/to/JiT
conda env create -f jit-torch/environment.yaml
conda activate jit
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`environment.yaml` 固定 PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.4、NumPy 1.22 和 JiT 指定的 `torch-fidelity`。

## Jittor 环境安装

```bash
conda create -n jit-jittor python=3.10.20 -y
conda activate jit-jittor
python -m pip install -r /path/to/JiT/jit-jittor/requirements.txt
```

Jittor GPU 运行需要 Linux、NVIDIA GPU、可用的 CUDA/cuDNN 开发文件和 C++ 编译器。macOS 本地仓库可用于代码管理和静态检查，不能复现 A100 CUDA 运行。

## amax 上的 Jittor CUDA/cuDNN 变量

amax 的系统 CUDA 为另一版本，仅激活 Conda 环境会让 Jittor 误选系统 CUDA，并可能报 `cudnn.h not found`。运行 Jittor 前必须指向已验证的 CUDA 12.2/cuDNN 8 共享工具链：

```bash
export JITTOR_PY=/home/xutianyi/miniconda3/envs/jit-jittor/bin/python
export JITTOR_CUDA_ROOT=/mnt/nfs/home/xutianyi/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux
export PATH="$JITTOR_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$JITTOR_CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$JITTOR_CUDA_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
export nvcc_path="$JITTOR_CUDA_ROOT/bin/nvcc"
export JITTOR_HOME="/tmp/${USER}_jit_cache"

"$JITTOR_PY" -c \
  "import jittor as jt; print(jt.__version__, jt.has_cuda)"
```

`JITTOR_HOME` 应使用每个用户、每个节点可写的本地缓存目录；不要复用其他用户的 Jittor cache。

## Jittor 1.3.11 A100 BF16 补丁

远程实现包含三个可重复执行的环境补丁，它们只修改当前用户拥有的 Conda 环境，不需要 `sudo`：

```bash
cd /path/to/JiT/jit-jittor
"$JITTOR_PY" patches/apply_jittor_bf16_cudnn_patch.py
"$JITTOR_PY" patches/apply_jittor_bf16_cudnn_training_patch.py
"$JITTOR_PY" patches/apply_jittor_mpi_world_size_patch.py
```

- 第一个补丁修正 BF16 cuDNN 前向卷积的 FP32 accumulation descriptor。
- 第二个补丁覆盖 BF16 卷积 backward-data 和 backward-filter。
- 第三个补丁修正该 Jittor 版本在 MPI world-size 下的兼容路径。

补丁应在每个新 Conda 环境中执行一次。如果补丁脚本报告目标源码模式不匹配，不要强行修改其他 Jittor 版本。

## 基础验证

```bash
# Torch
/home/xutianyi/miniconda3/envs/jit-torch/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"

# Jittor（先设置上述 CUDA/cuDNN 变量）
"$JITTOR_PY" -c \
  "import jittor as jt; jt.flags.use_cuda=1; print(jt.__version__, jt.has_cuda)"
```
