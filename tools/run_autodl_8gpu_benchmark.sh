#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT=/root/autodl-tmp/JiT/jit-jittor
ENV_ROOT=/root/autodl-tmp/JiT/envs/jit-jittor
RUN_ROOT=/root/autodl-tmp/JiT/evidence/jittor_8gpu_tar_benchmark_20260808
RUN_TAG=${RUN_TAG:-run_fp32_m16_a8_s16_r3}
MICRO_BATCH=${MICRO_BATCH:-16}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-8}
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-16}
RUN_DIR=${RUN_ROOT}/${RUN_TAG}
DATA_TAR=/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar
DATA_INDEX=/root/autodl-tmp/JiT/data-index/imagenet_train_first_class.npz

if [[ -e "${RUN_DIR}/started_at.txt" ]]; then
  echo "Refusing to overwrite existing run: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
date -Iseconds > "${RUN_DIR}/started_at.txt"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 1 > "${RUN_DIR}/gpu_global.csv" &
MONITOR_PID=$!

cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PATH=${ENV_ROOT}/bin:${PATH}
export JITTOR_HOME=/root/autodl-tmp/JiT/jittor-home/autodl-a800
export TMPDIR=/root/autodl-tmp/JiT/tmp
export nvcc_path=/root/autodl-tmp/JiT/cuda-overlay/bin/nvcc.wrapper
export LD_LIBRARY_PATH=${ENV_ROOT}/lib:/root/autodl-tmp/JiT/cuda-overlay/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMPI_MCA_opal_cuda_support=true
export nccl_include_path=/usr/include
export nccl_lib_path=/usr/lib/x86_64-linux-gnu

cd "${PROJECT_ROOT}"
set +e
mpirun --allow-run-as-root --bind-to none \
  --mca btl_vader_single_copy_mechanism none \
  -np 8 \
  -x PATH -x JITTOR_HOME -x TMPDIR -x nvcc_path -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES -x OMPI_MCA_opal_cuda_support \
  -x nccl_include_path -x nccl_lib_path \
  "${ENV_ROOT}/bin/python" train_benchmark.py \
  --data-tar "${DATA_TAR}" \
  --data-index "${DATA_INDEX}" \
  --output-dir "${RUN_DIR}" \
  --monitor-script "${PROJECT_ROOT}/tools/monitor_resources.py" \
  --image-size 256 \
  --precision fp32 \
  --micro-batch "${MICRO_BATCH}" \
  --accumulation-steps "${ACCUMULATION_STEPS}" \
  --rounds "${ROUNDS}" \
  --steps-per-round "${STEPS_PER_ROUND}" \
  --num-workers 2 \
  --lr 0.0002 \
  --warmup-epochs 0 \
  --seed 0 \
  --log-every 4 \
  2>&1 | tee "${RUN_DIR}/mpirun.log"
RUN_RC=${PIPESTATUS[0]}
set -e

echo "${RUN_RC}" > "${RUN_DIR}/exit_code.txt"
date -Iseconds > "${RUN_DIR}/finished_at.txt"
exit "${RUN_RC}"
