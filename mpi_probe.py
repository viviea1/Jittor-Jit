#!/usr/bin/env python3
import json
import socket
import time

import jittor as jt


jt.flags.use_cuda = 1
rank = jt.compile_extern.mpi.world_rank() if jt.in_mpi else 0
world = jt.compile_extern.mpi.world_size() if jt.in_mpi else 1

started = time.perf_counter()
x = jt.ones((1024, 1024), dtype="float32") * float(rank + 1)
y = x.mpi_all_reduce("mean") if jt.in_mpi else x
y.sync()

print(
    json.dumps(
        {
            "host": socket.gethostname(),
            "rank": rank,
            "world_size": world,
            "in_mpi": jt.in_mpi,
            "has_mpi": jt.compile_extern.has_mpi,
            "nccl_loaded": jt.compile_extern.nccl_ops is not None,
            "mean": float(y.mean().item()),
            "seconds": time.perf_counter() - started,
        }
    ),
    flush=True,
)
