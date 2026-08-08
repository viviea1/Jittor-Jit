#!/usr/bin/env python3
"""Apply the Jittor 1.3.11 BF16 cuDNN forward-convolution fix.

The patch is local to the active Python environment and is idempotent.
No sudo permission is required when the conda environment belongs to the
current user.
"""

import importlib.util
from pathlib import Path


OLD = """\
        CUDNN_CROSS_CORRELATION, getDataType<Ty>()
"""

NEW = """\
        // cuDNN requires FP32 accumulation for BF16 convolution. Passing
        // CUDNN_DATA_BFLOAT16 here makes cudnnSetConvolutionNdDescriptor
        // return CUDNN_STATUS_BAD_PARAM on A100 with cuDNN 8.
        CUDNN_CROSS_CORRELATION,
        std::is_same<Ty, __nv_bfloat16>::value
            ? CUDNN_DATA_FLOAT
            : getDataType<Ty>()
"""


def main():
    spec = importlib.util.find_spec("jittor")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Jittor is not installed in this Python environment")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    target = (
        package_root
        / "extern"
        / "cuda"
        / "cudnn"
        / "ops"
        / "cudnn_conv_op.cc"
    )
    source = target.read_text(encoding="utf-8")

    if NEW in source:
        print(f"Already patched: {target}")
        return
    if OLD not in source:
        raise SystemExit(
            f"Expected Jittor 1.3.11 source pattern not found: {target}"
        )

    backup = target.with_suffix(".cc.before-bf16-fix")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()
