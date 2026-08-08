#!/usr/bin/env python3
"""Record GPU, process-tree CPU/RSS and host RAM once per second."""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def read_proc_table():
    table = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / "stat").read_text().split()
            pid = int(fields[0])
            table[pid] = {
                "ppid": int(fields[3]),
                "ticks": int(fields[13]) + int(fields[14]),
                "rss_bytes": int(fields[23]) * os.sysconf("SC_PAGE_SIZE"),
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return table


def descendants(root_pid, table):
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, values in table.items():
            if pid not in result and values["ppid"] in result:
                result.add(pid)
                changed = True
    return result


def read_pss(pid):
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return 0
    return 0


def read_system_cpu():
    values = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    numbers = [int(value) for value in values]
    idle = numbers[3] + numbers[4]
    return sum(numbers), idle


def read_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def read_gpu():
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,"
        "utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    output = subprocess.check_output(command, text=True).strip()
    return [float(value.strip()) for value in output.split(",")]


def main():
    args = parse_args()
    clock_ticks = os.sysconf("SC_CLK_TCK")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "elapsed_seconds",
        "process_count",
        "process_tree_cpu_percent",
        "process_tree_rss_bytes",
        "process_tree_pss_bytes",
        "host_cpu_percent",
        "host_memory_used_bytes",
        "host_memory_available_bytes",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "gpu_power_watts",
        "gpu_temperature_c",
    ]

    start = time.monotonic()
    previous_time = start
    previous_ticks = None
    previous_total_cpu, previous_idle_cpu = read_system_cpu()

    with output.open("w", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            table = read_proc_table()
            if args.pid not in table:
                break
            family = descendants(args.pid, table)
            current_ticks = sum(
                table[pid]["ticks"] for pid in family if pid in table
            )
            current_time = time.monotonic()
            interval = max(current_time - previous_time, 1e-9)
            if previous_ticks is None:
                process_cpu = 0.0
            else:
                process_cpu = (
                    (current_ticks - previous_ticks) / clock_ticks / interval * 100
                )
            process_rss = sum(
                table[pid]["rss_bytes"] for pid in family if pid in table
            )
            process_pss = sum(read_pss(pid) for pid in family)

            total_cpu, idle_cpu = read_system_cpu()
            total_delta = total_cpu - previous_total_cpu
            idle_delta = idle_cpu - previous_idle_cpu
            host_cpu = (
                100.0 * (1.0 - idle_delta / total_delta) if total_delta else 0.0
            )
            memory_total, memory_available = read_memory()
            try:
                gpu = read_gpu()
            except (subprocess.SubprocessError, ValueError):
                gpu = [float("nan")] * 6

            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "elapsed_seconds": current_time - start,
                    "process_count": len(family),
                    "process_tree_cpu_percent": process_cpu,
                    "process_tree_rss_bytes": process_rss,
                    "process_tree_pss_bytes": process_pss,
                    "host_cpu_percent": host_cpu,
                    "host_memory_used_bytes": memory_total - memory_available,
                    "host_memory_available_bytes": memory_available,
                    "gpu_memory_used_mib": gpu[0],
                    "gpu_memory_total_mib": gpu[1],
                    "gpu_utilization_percent": gpu[2],
                    "gpu_memory_utilization_percent": gpu[3],
                    "gpu_power_watts": gpu[4],
                    "gpu_temperature_c": gpu[5],
                }
            )

            previous_time = current_time
            previous_ticks = current_ticks
            previous_total_cpu = total_cpu
            previous_idle_cpu = idle_cpu
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
