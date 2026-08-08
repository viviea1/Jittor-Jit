#!/usr/bin/env python3
"""Indexed, extraction-free ImageNet reader for the official nested train tar.

The ILSVRC2012 training archive contains one uncompressed tar per synset.  The
index stores absolute byte offsets for every JPEG, so workers can use
``os.pread`` without extracting images or sharing a mutable file position.
Class names and image names are sorted exactly like ImageFolder.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


BLOCK_SIZE = 512


def _tar_size(field: bytes) -> int:
    """Decode a POSIX/GNU tar size field."""
    if field and field[0] & 0x80:
        return int.from_bytes(field, "big", signed=True)
    text = field.rstrip(b"\0 ").lstrip(b" ")
    return int(text or b"0", 8)


def _tar_name(header: bytes) -> str:
    name = header[:100].split(b"\0", 1)[0]
    prefix = header[345:500].split(b"\0", 1)[0]
    value = prefix + (b"/" if prefix else b"") + name
    return value.decode("utf-8", errors="surrogateescape")


def build_index(
    tar_path: str | Path,
    index_path: str | Path,
    max_classes: int | None = None,
) -> dict:
    tar_path = Path(tar_path).resolve()
    index_path = Path(index_path).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with tarfile.open(tar_path, "r:") as outer:
        class_members = sorted(
            (member for member in outer.getmembers() if member.name.endswith(".tar")),
            key=lambda member: member.name,
        )
    if max_classes is not None:
        if max_classes <= 0:
            raise ValueError("max_classes must be positive")
        class_members = class_members[:max_classes]

    classes = [Path(member.name).stem for member in class_members]
    offsets: list[int] = []
    sizes: list[int] = []
    labels: list[int] = []
    fd = os.open(tar_path, os.O_RDONLY)
    try:
        for label, member in enumerate(class_members):
            position = int(member.offset_data)
            end = position + int(member.size)
            entries: list[tuple[str, int, int]] = []
            while position + BLOCK_SIZE <= end:
                header = os.pread(fd, BLOCK_SIZE, position)
                if len(header) != BLOCK_SIZE or header == b"\0" * BLOCK_SIZE:
                    break
                name = _tar_name(header)
                size = _tar_size(header[124:136])
                data_offset = position + BLOCK_SIZE
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    entries.append((name, data_offset, size))
                position = data_offset + ((size + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
            entries.sort(key=lambda value: value[0])
            offsets.extend(value[1] for value in entries)
            sizes.extend(value[2] for value in entries)
            labels.extend([label] * len(entries))
            if label == 0 or (label + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "index_progress",
                            "classes": label + 1,
                            "images": len(offsets),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        os.close(fd)

    class_blob = ("\n".join(classes) + "\n").encode()
    arrays = {
        "offsets": np.asarray(offsets, dtype=np.uint64),
        "sizes": np.asarray(sizes, dtype=np.uint32),
        "labels": np.asarray(labels, dtype=np.uint16),
        "classes": np.asarray(classes, dtype="U16"),
    }
    np.savez(index_path, **arrays)
    metadata = {
        "schema_version": 1,
        "tar_path": str(tar_path),
        "tar_bytes": tar_path.stat().st_size,
        "classes": len(classes),
        "images": len(offsets),
        "class_list_sha256": hashlib.sha256(class_blob).hexdigest(),
        "index_path": str(index_path),
        "index_bytes": index_path.stat().st_size,
        "seconds": time.perf_counter() - started,
    }
    index_path.with_suffix(index_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "index_complete", **metadata}), flush=True)
    return metadata


try:
    from jittor.dataset import Dataset
except ImportError:  # Index construction should also work without Jittor.
    Dataset = object


class NestedTarImageNet(Dataset):
    """Jittor dataset backed by an offset index into nested ImageNet tar."""

    def __init__(self, tar_path, index_path, transform=None):
        super().__init__()
        self.tar_path = str(Path(tar_path).resolve())
        self.index_path = str(Path(index_path).resolve())
        self.transform = transform
        with np.load(self.index_path, allow_pickle=False) as data:
            self.offsets = data["offsets"]
            self.sizes = data["sizes"]
            self.labels = data["labels"]
            self.classes = data["classes"].tolist()
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self._fd = None
        self._fd_pid = None
        self.set_attrs(total_len=int(self.offsets.shape[0]))

    def _get_fd(self):
        pid = os.getpid()
        if self._fd is None or self._fd_pid != pid:
            if self._fd is not None:
                os.close(self._fd)
            self._fd = os.open(self.tar_path, os.O_RDONLY)
            self._fd_pid = pid
        return self._fd

    def __getitem__(self, index):
        index = int(index)
        raw = os.pread(
            self._get_fd(), int(self.sizes[index]), int(self.offsets[index])
        )
        if len(raw) != int(self.sizes[index]):
            raise IOError(
                f"short tar read at index {index}: {len(raw)} != {int(self.sizes[index])}"
            )
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            value = self.transform(image) if self.transform else image.copy()
        return value, int(self.labels[index])

    def __del__(self):
        try:
            if self._fd is not None:
                os.close(self._fd)
        finally:
            try:
                super().__del__()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Build a deterministic prefix subset for throughput smoke tests.",
    )
    args = parser.parse_args()
    build_index(args.tar, args.output, max_classes=args.max_classes)


if __name__ == "__main__":
    main()
