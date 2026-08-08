# JiT

This repository is a Jittor port of [JiT](https://github.com/LTH14/JiT). The `/jit-torch` directory contains the reference version, while `/jit-jittor` contains the ported version.

## PyTorch environment

On a Linux/NVIDIA machine:

```bash
cd /path/to/JiT
conda env create -f jit-torch/environment.yaml
conda activate jit
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## Jittor environment

```bash
conda create -n jit-jittor python=3.10.20 -y
conda activate jit-jittor
python -m pip install -r /path/to/JiT/jit-jittor/requirements.txt
```

PyTorch and Jittor commands are provided in [RUNNING.md](RUNNING.md).
