#!/usr/bin/env python3
"""Patch Jittor 1.3.11 cuDNN 2-D convolution backward ops for BF16.

Recent cuDNN versions require FP32 compute/accumulation for BF16 convolution
descriptors.  The local inference patch already covers the forward operator;
training additionally needs backward-data and backward-filter operators.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import jittor


OLD = "CUDNN_CROSS_CORRELATION, getDataType<Ty>()"
NEW = """CUDNN_CROSS_CORRELATION,
                std::is_same<Ty, __nv_bfloat16>::value
                    ? CUDNN_DATA_FLOAT
                    : getDataType<Ty>()"""

RELATIVE_PATHS = (
    "extern/cuda/cudnn/ops/cudnn_conv_backward_x_op.cc",
    "extern/cuda/cudnn/ops/cudnn_conv_backward_w_op.cc",
)


def main() -> None:
    package_dir = Path(jittor.__file__).resolve().parent
    for relative_path in RELATIVE_PATHS:
        source = package_dir / relative_path
        text = source.read_text()

        if NEW in text:
            print(f"already patched: {source}")
            continue
        if text.count(OLD) != 1:
            raise RuntimeError(
                f"expected exactly one unpatched descriptor in {source}, "
                f"found {text.count(OLD)}"
            )

        backup = source.with_suffix(source.suffix + ".before-bf16-train-fix")
        if not backup.exists():
            shutil.copy2(source, backup)
        source.write_text(text.replace(OLD, NEW, 1))
        print(f"patched: {source}")
        print(f"backup:  {backup}")


if __name__ == "__main__":
    main()
