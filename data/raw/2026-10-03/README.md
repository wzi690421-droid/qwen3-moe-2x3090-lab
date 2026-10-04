# 在两张 RTX 3090 上独立研究 Qwen3 MoE：完整实验计划

编写日期：2026-10-02。对象：同一台机器上的两张 RTX 3090 24 GiB。

选定模型：**Qwen/Qwen3-30B-A3B-Instruct-2507-FP8**。

本手册提供命令、工具和判断方法；GPU 实验由你操作。随附脚本经过本机静态检查，尚未在你的租赁机器上完成 GPU 验证。下面的参数是待实测的起始配置，性能数字全部由你测量填写。

## 1. 完成这次实验，你应该得到什么

你要回答五个工程问题：

1. 这套官方权重能否在双 3090 上可靠地提供文本服务？实际用了什么量化 kernel？
2. 在固定输入、输出和版本后，并发 1/2/4/8 的吞吐与延迟如何变化？
3. 注意力仍使用 TP=2，把专家从 TP 切分改成 EP=2，是否改善你的工作负载？
4. profiler 中最耗时的计算或通信在哪里？能否对应到具体源码？
5. 根据证据做一个改动，收益能否在 A → B → A 中重现？

交付物是：可重启的服务配置、原始 JSON/日志、性能汇总 CSV、一份报告、一份你亲手完成的 CPU MoE 参考实现和代码链路笔记。最终贡献可以是有复验数据的文档，也可以是解决具体问题的代码；先有证据，再决定 PR 内容。

**“官方”在这里有明确范围：**模型是 Qwen 官方发布，模型卡给出了 vLLM 使用说明，vLLM 有对应架构和 FP8 Marlin 路径。这并不等于上游公开认证了“这个版本＋这台双 3090＋这些参数”，也没有可直接套用的官方双 3090 tok/s 合格线。[官方模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)

## 2. 为什么这个模型适合这个实验

| 项目 | 配置与意义 |
|---|---|
| 架构 | Qwen3MoeForCausalLM，纯文本，自回归生成 |
| 参数 | 总参数约 30.5B，每 token 激活约 3.3B |
| 层数 | 48 |
| 专家 | 每个 MoE 层 128 个专家，每 token 选 8 个 |
| 隐藏维度 | 2048；专家 intermediate_size 为 768 |
| 模式 | 仅 non-thinking；不需要额外设置 enable_thinking=False |
| 权重 | 官方 E4M3 FP8，128×128 block scale，部分层保留高精度 |
| 固定 revision | `5a5a776300a41aaa681dd7ff0106608ef2bc90db` |
| 权重文件 | 4 个 safetensors 分片，总计 31,175,618,584 bytes，约 29.03 GiB |

参数来自官方模型卡与固定 revision 的 config；分片大小来自 [Hugging Face 模型元数据](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8?blobs=true)。下载脚本会重新验证 revision、文件大小和官方 LFS SHA256。

**激活 3.3B 不表示只需存储 3.3B 权重。**不同 token 会选择不同专家，本实验把完整权重放在两张 GPU 上。磁盘大小也不等于运行时显存：加载时会重排权重，并分配缓存、激活、workspace、CUDA graph 和通信缓冲区。

3090 是 Ampere。这里使用的 FP8 路径是 **W8A16：压缩的 8 位权重＋16 位激活**，通过 Marlin 计算；不能把结果称为 3090 原生 FP8 Tensor Core 性能。普通 Linear 与 MoE 专家有各自的 backend，启动时两者都要核对。[FP8 文档](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)、[固定版本 Marlin MoE 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py)

## 3. 时间、预算与实验顺序

以下是排期估计，不是实测承诺。**模型和环境已就绪后，核心部分预计 1.5～3 小时 GPU 时间；全部扩展再留 0.5～1.5 小时。**CPU 练习和报告整理可以不租 GPU。

| 阶段 | 做什么 | 预计时间 | 是否必须 |
|---|---|---:|---|
| P0 | 阅读模型配置，完成 CPU MoE 小练习，准备文件 | 本机 2～4 小时 | 学习必须；无需占 GPU |
| P1 | 硬件、驱动、安装与 NCCL 求和检查 | 20～40 分钟 | 必须 |
| P2 | 下载、SHA256、准备固定输入 | 下载单独估算，校验 2～10 分钟 | 必须 |
| P3 | TP=2 启动、健康、20 条功能检查 | 10～20 分钟 | 必须 |
| P4 | TP 基线：固定长度＋长输入＋真实文本 | 15～35 分钟 | 必须 |
| P5 | EP=2：功能检查＋相同矩阵 | 20～40 分钟 | 核心对照；不支持时按下文退出 |
| P6 | profiler＋选一项优化＋A/B/A | 20～45 分钟 | 必须做定位；优化按证据选择 |
| P7 | NCCL 默认路径 / 禁用 P2P 对照 | 10～30 分钟 | 通信扩展，可第二次做 |
| P8 | 汇总、备份、停止计费 | 10～20 分钟 | 必须 |

下载约 31.2 GB：20 MB/s 约 26 分钟，10 MB/s 约 52 分钟，5 MB/s 约 104 分钟。只估算权重，不含安装包。开跑 5～10 分钟后按实际速度重算，下载时长明显超预算就停止继续空转。平台若支持关 GPU 后保留持久盘/CPU 下载，可使用该功能；先确认磁盘是否真的共享和持久。

租赁前确认的是**两张卡整机总价**。总费用 = 实际计费小时 × 整机单价；若为 10 元/小时，3 小时为 30 元。首次建议给 GPU 会话设 3 小时提醒，到点保存证据并决定是否延长；不要把完整实验理解成无限排错。

执行顺序：`P0 → P1 → P2 → P3 → P4 → P5 → P6 → P8`。P7 在理解主线后增加，所有 GPU 组串行进行。

## 4. P0：租机前，先弄懂一个 MoE 层

本节你亲手写代码。随附测量脚本是工具，不代表你已掌握其实现。

先创建 `notes/toy_moe.py` 和 `notes/code-path.md`，使用 CPU float32、固定随机种子，设置：tokens=4、hidden=8、intermediate=16、experts=8、top_k=2。模拟张量与计算：

```text
x: [4, 8]
router_weight: [8, 8]
router_logits = x @ router_weight            → [4, 8]
scores = softmax(router_logits, dim=-1)
topk_weights, topk_ids = topk(scores, k=2)    → 两个 [4, 2]
topk_weights /= topk_weights.sum(-1, keepdim=True)

每个专家 e 有三个独立矩阵：W_gate_e、W_up_e、W_down_e
expert_e(x) = (SiLU(x @ W_gate_e) * (x @ W_up_e)) @ W_down_e
output[token] = Σ topk_weight[token,j] * expert_topk_id[token,j](x[token])
```

写两种实现：先按 token 循环计算；再把分配给同一专家的 token 收集起来批量计算并散回输出。固定同一套权重，用 `torch.allclose(..., atol=1e-5, rtol=1e-5)` 比较。这个容差是 CPU 小练习的起点，不是完整 FP8 模型的精度门槛。

你要打印：每个 token 选中的专家、权重和每个专家接到的 token 数。注意总路由次数是 4×2=8，一个 token 可以出现于两个专家组。

完成标准：你能解释“专家是参数不同的 FFN”“router 处理的是当前层的隐藏向量”“为什么需要 token 分组”“为什么最后要加权合并”。不要先改真实模型的 top-k；那会改变模型计算与质量。

## 5. 租机要求与目录

### 5.1 选择机器

- 同机独占 2×RTX 3090 24 GiB，不能是一张 3090 加另一种卡。
- Linux x86_64，优先 Ubuntu 22.04/24.04，glibc ≥2.28。
- Python 3.11 或 3.12；本计划固定 vLLM 0.30.0 的 CUDA 12.9 wheel。
- **本手册采用驱动 ≥575.57.08 作为软件栈预检条件**，优先 R580 或更新。CUDA minor compatibility 存在更低门槛，但首次实验不依赖它绕过 PTX/JIT 差异。若平台只提供 R570/R535，这套安装命令先不要执行，需要重新选择匹配的软件栈和重新冻结实验。
- 建议 RAM ≥64 GiB，持久盘空闲 ≥100 GiB；源码、venv、安装缓存和 trace 也占空间。
- NVLink 桥是研究 NVLink 的硬件前提，**不是完成 TP/EP 部署的必需条件**。没有桥也能继续做 PCIe/P2P 通信实验，但报告注明实际拓扑。

版本固定的原因：截至编写日期，上游发布 v0.30.0 并提供 cu129 构建；该版本 Marlin MoE 源码提供 Ampere 所需的路径。旧版 FP8 Marlin 的 all-to-all 支持不同，不能拿旧版 API 直接替换本手册。[v0.30.0 发布与构建](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)、[CUDA 12.9 驱动对应表](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)

### 5.2 上传本手册包

本机目录为 `/home/hean/AI infra/experiments/qwen3-moe-2x3090`。使用平台文件上传，或在本机执行下面命令，把 `YOUR_SSH_PORT`、`YOUR_SSH_HOST` 替换为**本次实例**的信息：

```bash
scp -P YOUR_SSH_PORT -r "/home/hean/AI infra/experiments/qwen3-moe-2x3090" \
  root@YOUR_SSH_HOST:/root/autodl-tmp/
```

`/root/autodl-tmp` 只是 AutoDL 的目录示例。使用其他平台时，先选该平台确认会保留的目录。本实验目录将同时保存模型和证据。

远端打开两个终端：**A 运行前台服务，B 发送请求和测量**。每个终端都执行：

```bash
cd /root/autodl-tmp/qwen3-moe-2x3090
source ./env.sh
set -o pipefail
mkdir -p artifacts runs notes models
```

目录用途：

```text
env.sh                  固定模型与端口；每个终端都 source
scripts/                下载、启动、检查、测量和汇总工具
workloads/              功能题、固定 token 输入、真实文本、数据哈希
artifacts/              硬件与软件快照、下载 manifest
runs/<run-name>/        每次启动独立目录；命令、日志、检查、bench、GPU 数据
notes/                  你写的 CPU MoE、代码链路和分析
models/                 官方模型文件，备份实验时可排除
REPORT_TEMPLATE.md      复制后填写实测报告
```

## 6. P1：先验证硬件与软件

### 6.1 硬件快照

```bash
nvidia-smi | tee artifacts/nvidia-smi.txt
nvidia-smi -L | tee artifacts/gpu-list.txt
nvidia-smi topo -m | tee artifacts/topology.txt
nvidia-smi nvlink --status > artifacts/nvlink-status.txt 2>&1
uname -a > artifacts/kernel.txt
ldd --version > artifacts/glibc.txt
free -h > artifacts/ram.txt
df -h "$LAB_DIR" > artifacts/disk.txt
python3 --version > artifacts/python-before.txt
```

判断方法：

- GPU0 ↔ GPU1 的 `NV#` 表示拓扑中有对应 NVLink 链路；同时查看 link status 是否 active。
- `PIX/PXB/PHB/SYS` 表示 PCIe/主机拓扑路径。它们不能单独证明 P2P 数据传输是否可用。
- 查询命令在平台受限制时报错，把输出保留下来；不要把“查询失败”写成“没有 NVLink”。
- 确认没有其他用户任务占用两张 GPU。不要终止不属于本实验的进程。
- `nvidia-smi` 顶部 CUDA Version 是驱动支持信息，不是你安装的 PyTorch runtime；安装后还要看 `torch.version.cuda`。

### 6.2 安装固定版本

先核对上述驱动和 RAM/磁盘要求。使用新的虚拟环境，让 vLLM 带入匹配的 PyTorch；不要预先任意安装 torch，也不要在压测之间升级包。[官方安装说明](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

```bash
python3 -m venv "$LAB_DIR/.venv"
source "$LAB_DIR/.venv/bin/activate"
python -m pip install --upgrade pip uv

VLLM_WHEEL_URL='https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
uv pip install "$VLLM_WHEEL_URL" \
  --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install pandas datasets

python -m pip freeze > artifacts/pip-freeze.txt
vllm serve --help=all > artifacts/serve-help.txt
vllm bench serve --help > artifacts/bench-help.txt
```

如果 `venv` 不可用，选提供 Python venv 的平台镜像，或安装发行版的 `python3-venv` 包。首次安装连续排错超过 20 分钟，保留错误并结束本轮安装尝试；不要开始源码编译来填补不匹配的 wheel。

安装后执行：

```bash
python - <<'PY' | tee artifacts/torch-gpu-check.txt
import importlib.metadata as m
import torch
print('vllm:', m.version('vllm'))
print('torch:', torch.__version__, 'runtime CUDA:', torch.version.cuda)
print('NCCL:', torch.cuda.nccl.version())
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
assert m.version('vllm').split('+')[0] == '0.30.0'
assert torch.version.cuda.startswith('12.9'), 'Stop: CUDA wheel stack differs from this plan'
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, 'GiB:', p.total_memory / 2**30, 'capability:', torch.cuda.get_device_capability(i))
    assert '3090' in p.name and torch.cuda.get_device_capability(i) == (8, 6)
print('peer-access API:', torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0))
for i in range(2):
    x = torch.ones((64,64), device=f'cuda:{i}', dtype=torch.bfloat16)
    assert (x @ x).float().mean().item() == 64
print('BF16 matmul OK')
PY
```

再做一个很小的真实 NCCL 检查；此时服务尚未启动：

```bash
timeout --signal=INT --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node=2 scripts/nccl_probe.py \
  --iterations 10 --out artifacts/nccl-preflight.json
```

通过条件：两个设备可用，BF16 运算成功，NCCL 所有 size 的求和结果正确。peer-access 返回 False 不直接禁止部署，NCCL 可使用其他路径，但先检查实际运行是否成功。NCCL 卡住超过 3 分钟则停止，记录拓扑和进程信息；这比先下载 31 GB 再发现通信有问题更省费用。

## 7. P2：下载、校验、固定输入

```bash
source "$LAB_DIR/.venv/bin/activate"
python scripts/download_model.py 2>&1 | tee artifacts/download.log
python scripts/prepare_workloads.py
```

下载使用固定 commit、4 个并发文件任务、可续传的 local_dir；只有一个下载进程。中断后重跑同一条命令。网络无法访问 Hugging Face 时，可在**运行 Python 前**设置平台提供的可信 HF 兼容 endpoint，例如 `export HF_ENDPOINT=https://hf-mirror.com`，仍使用相同 commit 并做 SHA256 校验；镜像不是官方精度或性能认证。

离线复核：

```bash
python scripts/download_model.py --verify-only
```

检查 `artifacts/model-manifest.json` 和 `workloads/hashes.json`。脚本生成三类输入：

- 20 个固定功能题：10 个自动精确/JSON 检查、10 个开放题人工查看。
- 功能题经过一次 chat template 后保存 **prompt token IDs**；所有 profile 使用同一份文件。
- 32 条不同的工程文本用于真实文本压测；它们是小型学习数据集，不代表完整生产流量。

固定长度随机 token 压测将由官方 bench 生成，固定 seed、长度和无共享固定前缀。随机数据用于控制形状；不能拿它评估回答质量或代表真实专家负载。

## 8. P3：启动 TP=2，建立功能基线

### 8.1 终端 A：启动

```bash
source ./env.sh
source "$LAB_DIR/.venv/bin/activate"
RUN_DIR="$LAB_DIR/runs/tp-$(date -u +%Y%m%dT%H%M%SZ)"
echo "$RUN_DIR"
bash scripts/launch.sh tp "$RUN_DIR"
```

这是前台服务。将打印的 `RUN_DIR` 复制到终端 B；不要同时启动第二套模型。

脚本中的起始配置：

| 参数 | 起始值 | 为什么 |
|---|---:|---|
| tensor-parallel-size | 2 | 单张 24 GiB 装不下这份 FP8 权重 |
| data-parallel-size | 1 | 一套逻辑服务，不启动两个完整模型副本 |
| dtype | bfloat16 | 非量化部分与相关激活使用 BF16，权重按 checkpoint 的 FP8 配置加载 |
| kv-cache-dtype | auto | 首轮保持模型对应的常规 KV 精度，避免同时引入 FP8 KV 变量 |
| gpu-memory-utilization | 0.85 | 起始预算；加载与预热后的可用空间需实测 |
| max-model-len | 8192 | 首轮不把原生 262K 长上下文当部署目标 |
| max-num-seqs | 8 | 明确服务调度上限；更高客户端并发可能排队 |
| max-num-batched-tokens | 2048 | 固定 prefill 调度预算，启用 chunked prefill |
| prefix caching | 关闭 | 避免重复请求缓存命中掩盖计算成本 |
| 执行模式 | 默认图/编译配置 | 先保存默认执行模式的实测基线 |

完整命令保存在该 run 的 `command.sh`，软件和实验变量在 `run-meta.json`。所有后续组沿用这些参数，除了明确指定的实验因素。

### 8.2 终端 B：就绪与 kernel 核对

```bash
source ./env.sh
source "$LAB_DIR/.venv/bin/activate"
RUN_DIR='把终端A打印的绝对目录粘贴到这里'
curl --fail --silent --show-error "$BASE_URL/health"
curl --fail --silent --show-error "$BASE_URL/v1/models" | tee "$RUN_DIR/models.json"
rg -n -i 'marlin|fp8|moe|kernel|weight|cache|graph|error|out of memory' "$RUN_DIR/server.log"
```

启动前几分钟健康请求暂时失败可以等待；**10 分钟仍未 ready，停止并按排错表判断**。至少保存：每卡权重占用、KV 容量、所选 Linear kernel、所选 MoE kernel、图捕获或禁用信息。不能只看到 `quantization=fp8` 就认定计算路径正确。

### 8.3 功能与重复性检查

```bash
python scripts/check_correctness.py --out "$RUN_DIR/correctness-before.json"
bash scripts/bench_matrix.sh "$RUN_DIR" smoke
```

20 个题按固定顺序串行，每题运行 3 次。自动题与开放题分开：

- 自动任务期望 10/10；格式没按要求输出时，先区分格式失败与语义错误，不直接认定 kernel 错。
- 开放题查看代码可执行性、翻译和事实是否正确，不能只看非空输出。
- 相同 profile 的逐 token 重复性期望 20/20。输出保存了真正的生成 token IDs 和 logprobs，不能把文本重新 tokenize 当成原始 IDs。
- `temperature=0` 和 seed 不保证跨 backend、跨并行策略逐 token 完全相同。

工具退出码 2 表示需要检查的诊断结果，**不等于证明模型有缺陷**。若同配置重复性不一致，先确认请求/输入哈希、串行执行、服务版本和其他进程；查看首个分叉位置的候选概率。排查限制 10 分钟，之后把该组标为“重复性未通过”；性能数据可以作为探索记录，不能宣称精度已经确认。

这些 20 题是回归烟测，不是 FP8 相对 BF16 的完整质量认证。它们也不能证明谁的浮点结果更接近数学真值。

## 9. P4：测量 TP 基线

### 9.1 先理解四个指标

- **TTFT**：发出请求到首个输出 token 的时间，包括服务端等待、prefill 和相关处理。
- **TPOT**：请求首 token 之后的平均每 token 时间；接近 decode 体验，单位 ms/token。
- **输出吞吐**：所有成功请求实际输出 token 总数 / 整个测量窗口时间。包含 prefill、调度和服务开销，不能直接称为纯 kernel decode 吞吐。
- **E2EL**：完整请求的端到端时间。客户端 `max-concurrency` 会限制在途请求，不等于 `max-num-seqs`。

### 9.2 完整命令

```bash
bash scripts/bench_matrix.sh "$RUN_DIR" main
bash scripts/bench_matrix.sh "$RUN_DIR" long
bash scripts/bench_matrix.sh "$RUN_DIR" real
python scripts/check_correctness.py \
  --out "$RUN_DIR/correctness-after.json" \
  --baseline "$RUN_DIR/correctness-before.json"
```

| suite | 工作负载 | 并发 | 重复 |
|---|---|---|---|
| main | 固定 1024 输入 / 256 输出，ignore EOS | 1/2/4/8 | 两轮，第二轮反向顺序 |
| long | 固定 4096 输入 / 128 输出 | 1/4 | 一轮定位 prefill 变化 |
| real | 32 条不同真实文本，输出上限 256，允许自然 EOS | 1/4 | 两轮 |
| burst，可选 | 1024 输入 / 128 输出，客户端超过 8 个调度槽 | 16 | 一轮，研究排队 |

main 请求数是 `max(16, 8×并发)`，每组预热 2 个请求；原始结果同时保存实际输入/输出长度。每个单项 10 分钟 timeout，任何错误或超时都会中断该 suite；不把没有完成的测试写成通过。

**为什么有两轮：**估计重复误差并发现温度、时钟或缓存热状态的影响。它是短矩阵重复，不是两小时稳定性烧机。按同一个并发分别记录两轮结果；差异超过 5% 先核对 GPU 温度、时钟、功率和请求统计，只补测受影响的点。

小样本的 P95 只作为观察；不把 16～64 个请求的尾延迟当成生产 SLA 认证。真实文本与随机形状数据分开报告，真实文本的自然输出长度不同，不能只用总用时比较速度。

每个 suite 保存官方 JSON、详细输出、完整客户端命令、1 秒 GPU 采样和压测前后 `/metrics`。`nvidia-smi memory.used` 高可能来自预分配，不表示 KV token 当前占用高；active/queued requests 和 KV 使用率要结合服务 metrics 的实际字段读取，不能把总显存当活跃缓存利用率。[官方 bench CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)

## 10. P5：只改变专家并行策略

先保存 TP 的 `RUN_DIR` 为 `TP_RUN_DIR`。在终端 A 按 Ctrl+C 停止旧服务，确认 `/health` 不再成功、端口已释放、两卡上没有该服务残留 worker。进程仍在时先确认 PID 与命令归属，针对自己的进程退出，不使用全局 `pkill python`。

然后在 A 启动新组：

```bash
RUN_DIR="$LAB_DIR/runs/ep-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/launch.sh ep "$RUN_DIR"
```

在 B 更新 `RUN_DIR`，把 `TP_RUN_DIR` 设置为此前的实际目录，健康检查成功后：

```bash
TP_RUN_DIR='之前TP组的绝对目录'
python scripts/check_correctness.py \
  --out "$RUN_DIR/correctness-before.json" \
  --baseline "$TP_RUN_DIR/correctness-before.json"
bash scripts/bench_matrix.sh "$RUN_DIR" smoke
bash scripts/bench_matrix.sh "$RUN_DIR" main
bash scripts/bench_matrix.sh "$RUN_DIR" long
bash scripts/bench_matrix.sh "$RUN_DIR" real
python scripts/check_correctness.py \
  --out "$RUN_DIR/correctness-after.json" \
  --baseline "$RUN_DIR/correctness-before.json"
```

两组的区别：

```text
TP 组：Attention TP=2；每个专家的矩阵也分片在两卡，计算部分结果后合并。
EP 组：Attention 仍 TP=2；MoE 按专家分布，两卡各保存对应专家，并分发/合并 token。
```

本例 DP=1，因此启用 EP 后 EP size=TP×DP=2。128 个专家均匀放置时每卡 64 个；应通过日志和源码核对实际映射。EP 使用通用 `allgather_reducescatter`，不要求 DeepEP 或数据中心专用 NVLink backend。[官方 EP 说明](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)

这是**并行策略**的受控对照，不是只比较两条硬件链路。EP 不保证快，尤其小 batch 的通信与分组开销可能抵消收益。

若 EP 仍出现“不支持 backend / quantization / parallel config”错误：保存完整日志，限制排错 15 分钟，停止该组。不要通过升级包、换 checkpoint、开 CPU offload 来悄悄改变对照。TP 基线、CPU MoE、profiler 和 NCCL 研究仍可独立完成；报告把 EP 记为该具体软件栈的未完成项。

若 TP 与 EP token 序列不同：记录首个不同位置，检查自动题与人工任务是否退化，并结合 logprobs/相同前缀分析。累加顺序变化可能影响 greedy argmax，不能仅凭分叉就发布“计算错误”的 Issue，也不能忽略实际任务质量下降。

## 11. P6：从数据走到代码，再选一个改动

### 11.1 把源码固定到与你运行的版本一致

可以在本机提前下载；远端下载时尽量不占 GPU 时间：

```bash
git clone --depth 1 --branch v0.30.0 \
  https://github.com/vllm-project/vllm.git "$LAB_DIR/upstream-vllm"
git -C "$LAB_DIR/upstream-vllm" rev-parse HEAD > artifacts/vllm-source-commit.txt
```

源码用于阅读；不要把它加到当前 `PYTHONPATH`，也不要在这个阶段替换已安装 wheel。

按下面顺序读，不需要通读整个仓库：

| 问题 | 源码入口/搜索词 | 必须写进笔记的内容 |
|---|---|---|
| 层如何连接 | `vllm/model_executor/models/qwen3_moe.py`，`Qwen3MoeDecoderLayer` | Attention、残差、norm、MoE 前后 shape |
| router/top-k 从哪来 | 同文件 `Qwen3MoeSparseMoeBlock`、`FusedMoEFactory`、`GateLinear` | 128、8、2048 从 config 进入何处 |
| 路由怎么执行 | `vllm/model_executor/layers/fused_moe/` 搜索 `topk_ids`、`renormalize` | 分数、专家 ID、分组、合并的位置 |
| 为什么选 Marlin | `layers/quantization/fp8.py`、`fused_moe/oracle/fp8.py` | 硬件 capability、量化粒度、候选 backend |
| 专家实际算什么 | `fused_moe/experts/marlin_moe.py` | token 排序、两次 GEMM、SiLU/mul、weighted reduce |
| C++/CUDA 入口 | `vllm/_custom_ops.py` 搜索 `moe_wna16_marlin_gemm`，然后在 `csrc/` 搜同名 | Python 参数如何传入算子；哪个维度被切分 |
| TP 通信 | `vllm/distributed/device_communicators/` | custom AllReduce 与 NCCL 的选择条件 |

有些版本把 gate 放进融合层，`router_logits` 参数名也未必代表已经计算好的 logits。必须跟到实际 gate/forward 计算，不能只根据变量名画调用链。

记录真实模型的一层：输入 `[T,2048]`，路由分数 `[T,128]`，top-k IDs/weights `[T,8]`；每个专家 gate/up/down 投影如何产生并组合结果。对照你在 P0 写的小实现，解释张量为什么按专家重排。

### 11.2 获取短 trace

停止当前服务后，启动单独的诊断组：

```bash
RUN_DIR="$LAB_DIR/runs/tp-profile-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/launch.sh tp-profile "$RUN_DIR"
```

另一终端健康检查成功后：

```bash
bash scripts/bench_matrix.sh "$RUN_DIR" profile
```

打开 trace 时使用 [Perfetto](https://ui.perfetto.dev/)，上传该 run 的 `traces/` 文件。只采两条 32-token 请求；不为看图采完整压测矩阵。诊断 trace 有额外开销，不参加正式吞吐对照。[官方 profiling 指南](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/contributing/profiling.md)

看图顺序：

1. 找到一次 prefill 和连续几个 decode step，比较同一 step 内两张 GPU 的时间线。
2. 看 GPU 是否有空洞，CPU launch/同步是否位于关键路径。
3. 标记 Attention、MoE GEMM、路由/重排、AllReduce/分发合并的区间。
4. 核对通信与计算是否重叠；不能把所有 kernel duration 直接相加当端到端时间。
5. 将最耗时的区间对应到上述文件和函数，写出可证伪的瓶颈假设。

只看到 GPU utilization 高不能证明算力满载；低 MFU 也不自动代表可以提高。decode 可能受权重/KV 访存、kernel 启动和通信限制。

### 11.3 第一个优化实验如何选

每次只选一行，不做参数全组合穷举：

| 观察 | 候选因素 | 验证方式 |
|---|---|---|
| EP 在高并发更快，功能检查没退化 | 采用 EP 并行策略 | 新 TP → 新 EP → 新 TP，各跑 main；重点复测 C=4/8 与 real C=4 |
| 小并发 CPU launch/同步明显 | 默认执行模式 vs `tp-eager` | eager 是解释性负对照；更慢也有价值，用来理解默认图/编译路径 |
| 长输入使 TTFT 明显增加，prefill 占主导 | `max-num-batched-tokens` 2048→4096 | 复制 launch 脚本到 notes，只改这一项；测 long 和 main，检查峰值显存 |
| 高并发等待明显，仍有容量与质量余量 | `max-num-seqs` 8→16 | 只改这一项；测试 C=8/16 与功能回归，不能承诺收益 |
| 主要时间落在一个 MoE kernel | 先提取对应形状做 microbenchmark | 先读现有测试与实现，再选代码改动；不直接改专家数/top-k |

`--enforce-eager` 可能同时关闭 CUDA graph 和编译优化，因此“默认 vs eager”是执行模式对照，不能把全部差值写成 CUDA graph 的独立收益。

采用改动的学习门槛：零请求错误，自动任务不退化，人工题没有明显质量退化；关键工作负载收益稳定超过重复噪声。比如 10% 可作为本次预先约定的目标，**不是上游规定的通用门槛**。吞吐改善但 TTFT/TPOT 变差时，说明你接受哪类负载的取舍。

每次优化执行 A → B → A；如果最后 A 明显偏离最初 A，优先检查温度、时钟、其他负载和测量输入。失败也交付完整数据与解释，不为了“有收益”不断追加变量。[官方优化说明](https://docs.vllm.ai/en/stable/configuration/optimization/)

## 12. P7：研究 NCCL 与 NVLink，避免混淆结论

### 12.1 先做集体通信对照

停止模型服务、核对 GPU 空闲后：

```bash
NCCL_DEBUG=INFO timeout --signal=INT --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node=2 scripts/nccl_probe.py \
  --out artifacts/nccl-default.json 2>&1 | tee artifacts/nccl-default.log

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 \
  timeout --signal=INT --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node=2 scripts/nccl_probe.py \
  --out artifacts/nccl-no-p2p.json 2>&1 | tee artifacts/nccl-no-p2p.log
```

看 payload 4 KiB / 64 KiB / 1 MiB / 8 MiB 的延迟、求和校验和 NCCL transport 日志。NCCL 日志可能只写 P2P、SHM 等 transport，是否真的跨 NVLink 还要结合拓扑和链路状态确认。

这里 `payload_GB_per_s=每rank payload/操作时间`，不是线缆带宽或 nvidia 规格里的双向带宽；也不是 nccl-tests 的所有 busbw 定义。需要更系统的集体通信测量可使用 [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)，纯设备复制可用 [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)，作为下一次扩展。

### 12.2 映射到模型服务

分别启动 `tp-nccl` 与 `tp-host`，每次新 run、完成正确性检查，按 main suite 或先 C=1/4/8 小范围测量：

```bash
# 终端 A，先运行一个 profile；测完 Ctrl+C，再启动另一个。
RUN_DIR="$LAB_DIR/runs/tp-nccl-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/launch.sh tp-nccl "$RUN_DIR"

# 下一次启动：
RUN_DIR="$LAB_DIR/runs/tp-host-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/launch.sh tp-host "$RUN_DIR"
```

这两组都关闭 vLLM custom AllReduce，区别是 NCCL 是否允许 P2P。`NCCL_P2P_DISABLE=1` 禁止 P2P，**没有把 NVLink 强制改成 PCIe P2P**；可能改用 SHM/经主机内存的路径。正确结论是“允许 P2P vs 禁止 P2P 的服务差异”，不能声称纯 NVLink 对 PCIe 的收益。[NCCL 环境变量定义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-disable)

原始 `tp` 组可能使用 custom AllReduce，单设 NCCL 环境变量不保证改变它的通信路径；这就是本扩展显式关闭它的原因。实际 backend 仍需日志和 trace 证据。

## 13. 出问题时的决策表

| 现象 | 先看什么 | 限定动作/下一步 |
|---|---|---|
| wheel 安装不匹配 | 驱动、Python、架构、glibc、torch CUDA、完整报错 | 20 分钟内修正镜像/依赖；不盲升级驱动或现场编译整个框架 |
| 下载慢或断开 | 实际速率、剩余容量、endpoint、是否有重复下载进程 | 同 commit 续传；重算费用；下载校验不通过不启动 |
| 权重加载阶段 OOM | 每 GPU weight/repack 日志、其他占用、加载峰值 | 分清权重/重排峰值，降低 context 通常不能解决权重本身过大 |
| 加载后缓存空间不足 | profile 的内存预算、KV token 数、运行时峰值 | 新诊断 run 先 context 8192→4096；一次只改一项 |
| 图捕获阶段 OOM | 捕获 batch、graph 内存、启动日志 | 新 `tp-eager` 诊断；若采用它，需要重新建立该模式基线 |
| 预热或 prefill OOM | input 长度、batched token 预算、激活峰值 | 新组把 batched tokens 2048→1024；不要同时增内存占比和并发 |
| NCCL/P2P 卡住 | topology、rank 日志、sum preflight、transport | timeout 停止自己任务；用禁 P2P 作诊断并明确路径代价 |
| EP 不支持 | 固定版本 Marlin/quant/parallel 支持，all2all backend | 15 分钟排错后保存证据退出 EP；不使用 DeepEP 硬凑 |
| greedy 有分叉 | 同输入、同参数、同 prefix 的 logits、数值差距 | 分开记录重复性和任务质量；不凭一个分叉宣布计算 bug |
| 吞吐变化 2～3% | 两轮噪声、GPU 时钟/温度、真实输出长度 | 只补受影响点；没有稳定证据就不声称优化成功 |

诊断成功后，必须把实际采用的配置作为一个新的基线重跑。不要把诊断数据拼进原基线。

## 14. P8：汇总、报告、归档、结束计费

```bash
python scripts/summarize.py "$LAB_DIR/runs" --out "$LAB_DIR/artifacts/results.csv"
cp REPORT_TEMPLATE.md notes/REPORT.md
```

填报告时至少给出：两轮各自数据、零错误证据、功能检查、启动配置、版本/数据哈希和实际拓扑。结果为负也要记录；没有测到的指标填“未测”，不填估计值。

每项收益手算一次：

```text
吞吐提升 = (B输出tok/s - A输出tok/s) / A输出tok/s
TTFT变化 = (B_TTFT - A_TTFT) / A_TTFT
两轮相对差异 = abs(轮1 - 轮2) / 两轮均值
```

按持续满载计算的粗略单位成本：`每百万输出token费用 = 整机元/小时 × 1,000,000 / (输出tok/s × 3600)`。它不包含下载空转、低利用率、输入成本、失败请求和 SLA 约束；只能在相同计费/负载口径下比较。

完成后先 Ctrl+C 停止自己的服务，再归档：

```bash
tar --exclude='./.venv' --exclude='./models' --exclude='./upstream-vllm' \
  -czf "$LAB_DIR/../qwen3-moe-3090-evidence.tar.gz" -C "$LAB_DIR" .
sha256sum "$LAB_DIR/../qwen3-moe-3090-evidence.tar.gz"
```

在本机下载证据包，例如：

```bash
scp -P YOUR_SSH_PORT root@YOUR_SSH_HOST:/root/autodl-tmp/qwen3-moe-3090-evidence.tar.gz \
  "/home/hean/AI infra/experiments/"
```

本机检查 SHA256 和 `tar -tzf`，确认 JSON、日志、报告和 trace 已收到，再在平台控制台停止/释放计费实例。停止 Python 服务不等于停止 GPU 计费；释放前确认模型盘是否保留。

## 15. 实验完成后，怎么走向一个有价值的 PR

第一轮先获得部署、对照和代码理解。不要把“用了已有启动参数”包装成新框架优化。

从测量中选择一个具体问题：明确形状下的 kernel 退化、backend 选择不合理、可复现的功能缺陷，或缺失且能实证补齐的部署文档。核对当前 upstream main 和已有 Issue/PR，确认没有已经解决。

拟改代码时，先提取小复现：输入 shape、dtype、硬件、配置、正确参考与失败/性能数据。然后最小改动、必要回归、重复测量和模型级复验。若要提交上游，最终在对应当前版本重新验证；本次 v0.30.0 的数据不能自动代表 main。

你能够不看脚本回答下面六题，就完成了本轮学习目标：

1. 为什么 3.3B 激活参数仍需要存储约 30.5B 总参数？
2. 为什么 FP8 权重能在没有原生 FP8 Tensor Core 的 3090 上运行？
3. TP 与 EP 分别切了什么，token 怎么到达对应专家？
4. 为什么总吞吐提高时单请求延迟可能变差？
5. 为什么禁用 P2P 不能直接得到 NVLink 对 PCIe 的独立结论？
6. 你这次优化改了什么代码/参数，哪个观测支持它，哪个实验能推翻它？

## 16. 独立执行完成检查

- [ ] 固定模型 revision、软件版本、完整命令和输入哈希。
- [ ] 拓扑与链路状态已记录；没有桥时如实注明。
- [ ] SHA256、GPU BF16 与 NCCL 求和检查通过。
- [ ] 确认实际 Linear 和 MoE backend，不只看模型名。
- [ ] TP 的功能题、冷/热重复性、固定与真实负载已保存。
- [ ] EP 对照已完成，或限制条件与失败证据已明确记录。
- [ ] profiler 中的关键区间能对应到具体源码。
- [ ] 完成至少一项有解释的对照；可获正收益，也可证伪假设。
- [ ] CPU MoE 参考实现由你完成，代码链路笔记能自己讲清楚。
- [ ] 报告区分实测、估算、假设、已证明内容和未测内容。
- [ ] 证据已回本机，控制台计费已停止。
