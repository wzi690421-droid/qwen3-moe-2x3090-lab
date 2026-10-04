#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
run_dir="${1:?Usage: bash scripts/bench_matrix.sh RUN_DIR smoke|main|long|real|profile|burst}"
suite="${2:?Provide a suite}"
[[ -f "$run_dir/run-meta.json" ]] || { echo "Missing server run metadata"; exit 2; }
[[ ! -e "$run_dir/bench/$suite" ]] || { echo "Suite already exists; preserve evidence"; exit 2; }
case "$suite" in
  smoke) concurrencies=(1); rounds=1; input_len=64; output_len=16 ;;
  main) concurrencies=(1 2 4 8); rounds=2; input_len=1024; output_len=256 ;;
  long) concurrencies=(1 4); rounds=1; input_len=4096; output_len=128 ;;
  real) concurrencies=(1 4); rounds=2; input_len=0; output_len=256 ;;
  profile) concurrencies=(1); rounds=1; input_len=1024; output_len=32 ;;
  profile8) concurrencies=(8); rounds=1; input_len=1024; output_len=32 ;;
  burst) concurrencies=(16); rounds=1; input_len=1024; output_len=128 ;;
  *) echo "Unknown suite: $suite"; exit 2 ;;
esac
curl --fail --silent --show-error "$BASE_URL/health" > /dev/null
result_dir="$run_dir/bench/$suite"
mkdir -p "$result_dir"
curl --fail --silent --show-error "$BASE_URL/metrics" > "$result_dir/metrics.before.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv --loop=1 > "$result_dir/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
profile="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$run_dir/run-meta.json")"
for ((round=1; round<=rounds; round++)); do
  order=("${concurrencies[@]}")
  # Reverse the second sweep so warm / hot ordering does not always favor C=8.
  if ((round == 2)); then
    order=()
    for ((i=${#concurrencies[@]}-1; i>=0; i--)); do order+=("${concurrencies[i]}"); done
  fi
  for concurrency in "${order[@]}"; do
    num_prompts=$((8 * concurrency))
    ((num_prompts >= 16)) || num_prompts=16
    [[ "$suite" != smoke ]] || num_prompts=4
    [[ "$suite" != profile ]] || num_prompts=2
    [[ "$suite" != profile8 ]] || num_prompts=8
    [[ "$suite" != long ]] || num_prompts=16
    [[ "$suite" != real ]] || num_prompts=32
    name="${suite}-i${input_len}-o${output_len}-c${concurrency}-r${round}"
    args=(vllm bench serve --backend vllm --base-url "$BASE_URL"
      --endpoint /v1/completions --model "$MODEL_DIR" --tokenizer "$MODEL_DIR"
      --served-model-name "$SERVED_MODEL"
      --num-prompts "$num_prompts" --num-warmups 2
      --max-concurrency "$concurrency" --request-rate inf --seed 20261002
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95
      --save-result --save-detailed --result-dir "$result_dir"
      --result-filename "$name.json" --label "$profile"
      --metadata "profile=$profile" "suite=$suite" "concurrency=$concurrency" "round=$round")
    if [[ "$suite" == real ]]; then
      args+=(--dataset-name custom --dataset-path "$LAB_DIR/workloads/real.jsonl"
             --custom-output-len "$output_len")
    else
      args+=(--dataset-name random --random-input-len "$input_len"
             --random-output-len "$output_len" --random-range-ratio 0
             --random-prefix-len 0 --ignore-eos)
    fi
    [[ "$suite" != profile && "$suite" != profile8 ]] || args+=(--profile)
    printf '%q ' "${args[@]}" > "$result_dir/$name.command.sh"
    printf '\n' >> "$result_dir/$name.command.sh"
    # This bound protects billed time; a timed-out suite is incomplete, never a pass.
    timeout --signal=INT --kill-after=20s 600s "${args[@]}" 2>&1 | tee "$result_dir/$name.log"
    python - "$result_dir/$name.json" "$num_prompts" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d['completed'] == int(sys.argv[2]), 'Incomplete benchmark; inspect errors'
assert d.get('failed', 0) == 0, 'Request errors; do not continue the matrix'
PY
  done
done
curl --fail --silent --show-error "$BASE_URL/metrics" > "$result_dir/metrics.after.txt"
echo "Finished suite: $result_dir"
