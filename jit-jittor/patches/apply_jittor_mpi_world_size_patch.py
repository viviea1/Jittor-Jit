#!/usr/bin/env python3
"""Fix Jittor 1.3.11's MPI hostname-gather buffer allocation.

The upstream source allocates ``hostHashs`` with ``mpi_world_rank`` entries.
Rank zero therefore gets a zero-length buffer before an all-gather writes one
entry per rank.  Allocate by ``mpi_world_size`` instead.  The patch is local to
the active user-owned conda environment and is idempotent.
"""

import importlib.util
from pathlib import Path


OLD = "uint64_t hostHashs[mpi_world_rank];"
NEW = "uint64_t hostHashs[mpi_world_size];"


def main():
    spec = importlib.util.find_spec("jittor")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Jittor is not installed in this Python environment")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    target = package_root / "extern" / "mpi" / "src" / "mpi_wrapper.cc"
    source = target.read_text(encoding="utf-8")

    if NEW in source:
        print(f"Already patched: {target}")
        return
    if OLD not in source:
        raise SystemExit(f"Expected Jittor 1.3.11 source pattern not found: {target}")

    backup = target.with_suffix(".cc.before-world-size-fix")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()
