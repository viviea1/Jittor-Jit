# 环境说明

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
