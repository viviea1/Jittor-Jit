#!/usr/bin/env python3
"""Validate the Jittor JiT-B/16 upstream-style random initialization."""

from __future__ import annotations

import hashlib
import json

import jittor as jt
import numpy as np

from model_jit import JiT_B_16


def tensor_digest(tensor) -> str:
    array = np.ascontiguousarray(tensor.numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    jt.flags.use_cuda = 1
    jt.set_global_seed(0)
    model = JiT_B_16(input_size=256, in_channels=3, num_classes=1000)
    model.pos_embed.stop_grad()
    jt.sync_all()

    parameters = dict(model.named_parameters())
    trainable = {
        name: parameter
        for name, parameter in parameters.items()
        if not parameter.is_stop_grad()
    }
    selected = {
        "blocks.0.attn.qkv.weight": parameters[
            "blocks.0.attn.qkv.weight"
        ],
        "x_embedder.proj1.weight": parameters[
            "x_embedder.proj1.weight"
        ],
        "y_embedder.embedding_table.weight": parameters[
            "y_embedder.embedding_table.weight"
        ],
        "pos_embed": parameters["pos_embed"],
    }

    result = {
        "jittor": jt.__version__,
        "seed": 0,
        "parameter_tensors": len(parameters),
        "parameter_count": sum(int(value.numel()) for value in parameters.values()),
        "trainable_parameter_count": sum(
            int(value.numel()) for value in trainable.values()
        ),
        "all_parameter_dtypes": sorted(
            {str(value.dtype) for value in parameters.values()}
        ),
        "digests": {
            name: tensor_digest(value) for name, value in selected.items()
        },
        "final_linear_abs_max": float(
            jt.abs(model.final_layer.linear.weight).max().item()
        ),
        "first_adaln_abs_max": float(
            jt.abs(model.blocks[0].adaLN_modulation[-1].weight).max().item()
        ),
        "in_context_posemb_std": float(
            model.in_context_posemb.float32().std().item()
        ),
        "label_embedding_std": float(
            model.y_embedder.embedding_table.weight.float32().std().item()
        ),
        "pos_embed_abs_max": float(
            jt.abs(model.pos_embed).max().item()
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
