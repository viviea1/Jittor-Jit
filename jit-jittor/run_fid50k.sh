#!/usr/bin/env bash
set -uo pipefail

OUT_DIR=/mnt/nfs/home/xutianyi/JiT/outputs/jittor-b16-256-fid50k
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JITTOR_PY=/home/xutianyi/miniconda3/envs/jit-jittor/bin/python
TORCH_PY=/home/xutianyi/miniconda3/envs/jit-torch/bin/python
CHECKPOINT=/mnt/nfs/home/xutianyi/JiT/checkpoints/checkpoint-last.pth
FID_STATS=/home/xutianyi/JiT/jit-torch/fid_stats/jit_in256_stats.npz
GPU_LOG="$OUT_DIR/gpu_usage.csv"
RUN_LOG="$OUT_DIR/run.log"
GEN_TIME_LOG="$OUT_DIR/generation_time.txt"
FID_TIME_LOG="$OUT_DIR/fid_time.txt"
STATUS_LOG="$OUT_DIR/status.txt"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR" || exit 1

monitor_pid=""
stop_monitor() {
    if [ -n "$monitor_pid" ] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap stop_monitor EXIT

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_watts\n' >"$GPU_LOG"
nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    -lms 1000 >>"$GPU_LOG" 2>&1 &
monitor_pid=$!

start_epoch=$(date +%s)
start_iso=$(date '+%Y-%m-%dT%H:%M:%S%z')

set +e
{
    echo "Starting Jittor BF16 50K generation at $start_iso"
    /usr/bin/time \
        -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kbytes=%M\nexit_status=%x' \
        -o "$GEN_TIME_LOG" \
        env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
        "$JITTOR_PY" fid50k.py \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUT_DIR" \
        --num-images 50000 \
        --batch-size 64 \
        --steps 50 \
        --cfg 3.0 \
        --interval-min 0.1 \
        --interval-max 1.0 \
        --seed 0 \
        --ema model_ema1 \
        --precision bf16 \
        --resume
    generation_status=$?

    if [ "$generation_status" -eq 0 ]; then
        echo "Starting JiT torch-fidelity evaluation"
        /usr/bin/time \
            -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kbytes=%M\nexit_status=%x' \
            -o "$FID_TIME_LOG" \
            env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
            "$TORCH_PY" evaluate_fid.py \
            --input-dir "$OUT_DIR/images" \
            --stats "$FID_STATS" \
            --output "$OUT_DIR/fid_results.json"
        evaluation_status=$?
    else
        evaluation_status=125
    fi

    end_epoch=$(date +%s)
    end_iso=$(date '+%Y-%m-%dT%H:%M:%S%z')
    {
        printf 'start=%s\n' "$start_iso"
        printf 'end=%s\n' "$end_iso"
        printf 'total_seconds=%s\n' "$((end_epoch - start_epoch))"
        printf 'generation_status=%s\n' "$generation_status"
        printf 'evaluation_status=%s\n' "$evaluation_status"
    } >"$STATUS_LOG"
    echo "Pipeline finished: generation=$generation_status evaluation=$evaluation_status"
    if [ "$generation_status" -ne 0 ]; then
        exit "$generation_status"
    fi
    exit "$evaluation_status"
} 2>&1 | stdbuf -oL awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush(); }' | tee -a "$RUN_LOG"
pipeline_status=${PIPESTATUS[0]}
set -e

stop_monitor
exit "$pipeline_status"
