# JiT

Code for training and evaluating JiT models on ImageNet.

## Repository layout

```text
jit-torch/
jit-jittor/
```

ImageNet data, checkpoints, generated images, and experiment outputs are not stored in Git.

## PyTorch environment

On a Linux/NVIDIA machine:

```bash
cd /path/to/JiT
conda env create -f jit-torch/environment.yaml
conda activate jit
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`environment.yaml` pins PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.4, NumPy 1.22, and the JiT-specific `torch-fidelity` package.

## Jittor environment

```bash
conda create -n jit-jittor python=3.10.20 -y
conda activate jit-jittor
python -m pip install -r /path/to/JiT/jit-jittor/requirements.txt
```

PyTorch commands are provided in [RUNNING.md](RUNNING.md).
