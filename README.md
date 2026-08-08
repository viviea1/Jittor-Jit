# JiT

Code for training and evaluating JiT models on ImageNet.

## PyTorch environment

On a Linux/NVIDIA machine:

```bash
cd /path/to/JiT
conda env create -f jit-torch/environment.yaml
conda activate jit
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

PyTorch commands are provided in [RUNNING.md](RUNNING.md).
