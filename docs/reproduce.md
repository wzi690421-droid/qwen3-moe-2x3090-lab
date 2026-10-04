# 复现这次实验

## 1 先离线复算已有数据

在仓库根目录执行：

```bash
python analysis/summarize_snapshot.py
```

推荐 Python 3.11；分析脚本只用标准库。完整 trace 已随原始材料保存，首次读取会有解压和 JSON 解析开销。

哈希核对：

```bash
cd data/raw/2026-10-03
sha256sum -c SHA256SUMS
```

## 2 准备 GPU 环境

本轮实测为同机双 RTX 3090、Python 3.12、vLLM 0.30.0、torch 2.13.0+cu130。完整软件清单见 `data/raw/2026-10-03/runs/tp-nccl-graph-20261003T075549Z/packages.txt`。需要兼容的驱动、约 29 GiB 权重磁盘空间，以及环境、缓存和日志的额外空间。

选择其他软件版本或机器时，保存新的环境信息与 run，不套用本轮性能结论。新环境可先按本轮 vLLM 版本安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'vllm==0.30.0'
python -m pip check
```

这组命令固定 vLLM 版本；它没有冻结全部传递依赖。安装后必须将 PyTorch、CUDA runtime、NCCL 与上述 `packages.txt` 比对，保存当前 `pip freeze`，并重新验证 GPU 和通信。若需要严格复现，应根据归档的软件清单重建环境。驱动兼容性不能由 `pip check` 判定。

```bash
source ./env.sh
mkdir -p artifacts
python -m pip freeze > artifacts/pip-freeze.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.nccl.version()); print(torch.cuda.can_device_access_peer(0, 1)); print(torch.cuda.can_device_access_peer(1, 0))'
```

这里的 P2P 检测与日志共同用于判断当前通信路径，不能单凭显卡名称认为 NVLink 可用。

## 3 下载和核对官方权重

```bash
source ./env.sh
set -o pipefail
mkdir -p artifacts
python scripts/download_model.py 2>&1 | tee artifacts/download-model.log
python scripts/download_model.py --verify-only
python scripts/prepare_workloads.py
```

下载固定 revision，核验四分片权重 SHA256，并检查模型配置。现有 workload 文件供阅读；运行准备脚本会按当前 tokenizer 重新生成输入。版本发生变化时重新记录 hashes。

## 4 建立 TP 服务基线

在终端 A 启动前台服务：

```bash
source ./env.sh
run_dir="$LAB_DIR/runs/tp-nccl-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/launch.sh tp-nccl "$run_dir"
```

等待服务 ready。终端 B 激活相同环境，设置相同的 run_dir，再进行检查：

```bash
source ./env.sh
curl --fail "$BASE_URL/health"
curl --fail "$BASE_URL/v1/models"
python scripts/check_correctness.py --repeats 5 --max-tokens 32 --out "$run_dir/correctness-5x-32.json"
bash scripts/bench_matrix.sh "$run_dir" smoke
bash scripts/bench_matrix.sh "$run_dir" main
```

正确性脚本发现任务失败或 token 序列不稳定时会退出 2，保留 JSON 供审查。应阅读 summary 和输出，记录判定后再决定是否继续；不将诊断退出码自动视为部署完全失败或完全通过。

每个 suite 都要求新的输出目录，避免覆盖历史证据。正式 main 为 1024 / 256，C=1/2/4/8，各两轮；其他 suite 的范围见脚本。关闭服务使用终端 A 的 Ctrl+C，并核对 GPU 进程退出。

EP 使用 `ep-nccl`；eager 使用 `tp-nccl-eager`。每个 profile 独占实例、使用新 run；正式对照保持其余参数相同。

## 5 获取阶段 trace

profile 启动入口为 `tp-nccl-profile` 或 `ep-nccl-profile`。先在新 run 启动服务，再在客户端分别执行：

```bash
bash scripts/bench_matrix.sh "$run_dir" profile
bash scripts/bench_matrix.sh "$run_dir" profile8
```

这里是 1024 / 32 的测量负载，不能混入 1024 / 256 的正式排名。CPU launch 时间与 GPU 执行时间也不能混用来划分阶段。

## 6 只复跑单卡 W13 基线

完整服务退出后，在同一已安装 vLLM 的环境中运行：

```bash
source ./env.sh
mkdir -p artifacts
CUDA_VISIBLE_DEVICES=0 python scripts/marlin_w13_probe.py   --model-dir "$MODEL_DIR" --seed 7   --output artifacts/marlin-w13-default-seed7-new.json
```

本轮只运行自动选择参数的默认配置。88.05 μs 是本机本次值，恢复环境后先确认数据可比，再讨论候选。该探针只测 W13，不含完整专家、模型或 NCCL。
