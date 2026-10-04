#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
profile="${1:?Usage: bash scripts/launch.sh PROFILE /absolute/run/directory}"
run_dir="${2:?Provide a fresh absolute run directory}"
[[ "$run_dir" = /* ]] || { echo "Run directory must be absolute"; exit 2; }
[[ ! -e "$run_dir" ]] || { echo "Use a NEW run directory: $run_dir"; exit 2; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "Download and verify the model first"; exit 2; }
# Control only this script's child environment; preserve the user's parent shell.
unset NCCL_P2P_DISABLE NCCL_ALGO NCCL_PROTO VLLM_BATCH_INVARIANT
export NCCL_DEBUG=INFO
extra=()
case "$profile" in
  tp) ;;
  ep) extra+=(--enable-expert-parallel) ;;
  ep-nccl) extra+=(--enable-expert-parallel --disable-custom-all-reduce) ;;
  tp-eager) extra+=(--enforce-eager) ;;
  tp-nccl) extra+=(--disable-custom-all-reduce) ;;
  tp-nccl-eager) extra+=(--disable-custom-all-reduce --enforce-eager) ;;
  tp-host)
    extra+=(--disable-custom-all-reduce)
    export NCCL_P2P_DISABLE=1
    ;;
  tp-auto-profile)
    mkdir -p "$run_dir/traces"
    extra+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run_dir/traces\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true}")
    ;;
  tp-nccl-profile)
    mkdir -p "$run_dir/traces"
    extra+=(--disable-custom-all-reduce --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run_dir/traces\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true}")
    ;;
  ep-nccl-profile)
    mkdir -p "$run_dir/traces"
    extra+=(--enable-expert-parallel --disable-custom-all-reduce --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run_dir/traces\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true}")
    ;;
  *) echo "Profiles: tp ep tp-auto-profile ep-nccl tp-eager tp-nccl tp-nccl-eager tp-host tp-nccl-profile ep-nccl-profile"; exit 2 ;;
esac
mkdir -p "$run_dir"
command=(vllm serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL"
  --tensor-parallel-size 2
  --data-parallel-size 1
  --distributed-executor-backend mp
  --all2all-backend allgather_reducescatter
  --dtype bfloat16
  --kv-cache-dtype auto
  --gpu-memory-utilization 0.85
  --max-model-len 8192
  --max-num-seqs 8
  --max-num-batched-tokens 2048
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --host 127.0.0.1
  --port "$SERVER_PORT"
  "${extra[@]}")
printf '%q ' "${command[@]}" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
python - "$profile" "$run_dir" <<'PY'
import importlib.metadata as m, json, os, sys
from pathlib import Path
import torch
keys = ["MODEL_ID", "MODEL_REVISION", "MODEL_DIR", "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_ALGO", "NCCL_PROTO"]
data = {"profile": sys.argv[1], "env": {k: os.environ.get(k) for k in keys},
        "packages": {k: m.version(k) for k in ["vllm", "torch", "transformers"]},
        "torch_cuda": torch.version.cuda}
Path(sys.argv[2], "run-meta.json").write_text(json.dumps(data, indent=2) + "\n")
PY
python -m pip freeze > "$run_dir/packages.txt"
nvidia-smi > "$run_dir/nvidia-smi.before.txt"
echo "RUN_DIR=$run_dir"
echo "Foreground server; Ctrl+C here stops this run."
"${command[@]}" 2>&1 | tee "$run_dir/server.log"
