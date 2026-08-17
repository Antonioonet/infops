#!/usr/bin/env bash
set -euo pipefail

ROOT="/app/workspace/infops"
BENCH="${ROOT}/infops_benchmark"
OUT="${ROOT}/benchmark_results"

mkdir -p "${OUT}/gadnr_auc_logs"

run_one() {
  local dataset="$1"
  local gpu="$2"
  local epochs="$3"
  local trials="$4"
  local seed="$5"
  local log="${OUT}/gadnr_auc_logs/${dataset}_gpu${gpu}.log"
  echo "=== ${dataset} gpu=${gpu} epochs=${epochs} trials=${trials} ==="
  python "${BENCH}/search_pygod_gadnr.py" \
    --dataset "${dataset}" \
    --objective auc \
    --gpu "${gpu}" \
    --seed "${seed}" \
    --epochs "${epochs}" \
    --batch-size 0 \
    --num-neigh -1 \
    --n-trials "${trials}" \
    --max-total-trials "${trials}" \
    --empty-cuda-cache \
    2>&1 | tee "${log}"
  python "${BENCH}/summarize_gadnr_best_auc.py"
}

(
  run_one russia 0 100 200 12121995
  run_one cuba 0 1 20 12122995
) > "${OUT}/gadnr_auc_logs/queue_gpu0.log" 2>&1 &
echo "gpu0_queue:$!"

(
  run_one venezuela 1 50 80 12121996
  run_one UAE 1 1 20 12122996
) > "${OUT}/gadnr_auc_logs/queue_gpu1.log" 2>&1 &
echo "gpu1_queue:$!"

(
  run_one iran 2 20 50 12121997
) > "${OUT}/gadnr_auc_logs/queue_gpu2.log" 2>&1 &
echo "gpu2_queue:$!"

(
  run_one china 3 20 50 12121998
) > "${OUT}/gadnr_auc_logs/queue_gpu3.log" 2>&1 &
echo "gpu3_queue:$!"
