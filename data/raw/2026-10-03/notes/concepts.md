# 概念速查：双 3090 上跑 Qwen3-30B-A3B-FP8 的瓶颈字典

所有数字来自本仓库已落盘的证据文件，不是推测。证据索引见文末。

## 表 1｜概念 → 含义 → 本机证据

| 概念 | 一句话解释 | 本机证据 |
|---|---|---|
| **算力-bound** | GPU 计算单元满载，瓶颈是"算得慢" | TTFT 默认 296 vs eager 304 ms（几乎不变）→ prefill 是算力-bound |
| **host-bound** | GPU 在等 CPU 派活，瓶颈是"派活慢" | TPOT 6.5 vs 57.7 ms（**7.7×**）→ decode 是 host-bound |
| **FP8** | 8 位浮点权重，显存/带宽减半 | 模型为 E4M3 FP8，128×128 block scale |
| **W8A16** | 权重 8 位存、激活 16 位算（weight-only） | sm86 无 FP8 硬件 → 只能这样 |
| **Marlin** | 专做"低精度权重 + 高精度激活"的融合 GEMM kernel | `Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod` |
| **MoE backend** | MoE 层的不同 kernel 实现方式 | `Using MARLIN Fp8 MoE backend out of potential backends: [...]` |
| **NVLink** | GPU 间专用直连（最快） | ❌ `all links are inActive` |
| **P2P** | 一卡直接读写另一卡显存 | ❌ `P2P 0→1: False` / `P2P 1→0: False` |
| **SHM** | 经主机内存中转（最慢的兜底路径） | ✅ `Channel 00 : 1[1] -> 0[0] via SHM/direct/direct` |
| **NUMA** | CPU 的内存域，跨域访问更慢 | GPU0 在 NUMA0、GPU1 在 NUMA1 |
| **SYS** | 最差的 GPU 间拓扑档位（跨 PCIe + 跨 NUMA） | `GPU0 ↔ GPU1 = SYS` |
| **all-reduce** | TP 下每层把两卡的 partial 结果求和 | 48 层 × 2 次 = **约 96 次/step** |

## 表 2｜想要的 vs 实际的（限制链）

| 环节 | 理想 | 本机实际 | 后果 |
|---|---|---|---|
| GPU 间通信 | NVLink / P2P | **SHM 跨 NUMA** | 每次通信两跳 PCIe + 一跳跨 NUMA |
| FP8 计算 | FP8 Tensor Core | **无（sm86）** | 只能 W8A16，先解压再算 |
| MoE kernel | DeepGEMM / FlashInfer-CUTLASS / TRTLLM | **MARLIN** | 兜底后端，kernel 数偏多 |
| 快速 all-reduce | FlashInfer AR / SymmMem / custom AR | **全部不可用** | 退化为纯 NCCL |
| 多卡通信原语 | NVLS multicast | **不可用** | 无多播加速 |

## 表 3｜性能数据（main 套件，i1024/o256）

| 并发 | 默认 TPOT | eager TPOT | 倍数 | 默认 tok/s | eager tok/s | 默认 TTFT |
|---|---|---|---|---|---|---|
| c1 | 6.5 ms | 57.7 ms | **7.7×** | 130.2 | 17.0 | 296 ms |
| c2 | 8.3 | 58.5 | 7.0× | 200.7 | 32.9 | 437 |
| c4 | 10.6 | 59.5 | 5.6× | 291.2 | 64.9 | 812 |
| c8 | 15.4 | 62.8 | 4.1× | 395.9 | 121.1 | 1242 |

结论：**TTFT 基本不变 + TPOT 差 7.7 倍** = decode 是 host-bound 的量化证据。

对 eager 解方程可进一步拆分：GPU 计算约 6.5 ms，CPU launch 约 51 ms，即默认路径下 decode 每步约 89% 的时间是"派活"而非"干活"。

## 表 4｜三个优化方向

| # | 方向 | 可证伪的假设 | 依据 | GPU 成本 | 可能产出 |
|---|---|---|---|---|---|
| ① | **EP=2 vs TP=2** | TP 每层 2 次 all-reduce；EP 只在 MoE 层通信 → EP 的 c1 TPOT 更优 | 约 96 次 SHM 通信与 TPOT 同量级 | 约 15 min | 报告 §3/§5 核心结论 |
| ② | **换 MoE backend** | kernel 更少的 backend 在 host-bound decode 下更快 | decode 被 kernel 数量支配 | 3 × 约 15 min | 可复现的后端推荐 → 文档/issue |
| ③ | **VLLM_BATCH_INVARIANT=1** | 开启后 greedy 的 token 分叉消失 | temp=0 仍分叉，且 `--enforce-eager` 未消除 | 约 5 min | 最小复现 + issue |

## 表 5｜MoE backend 候选

| backend | 实现特点 | 对 host-bound decode 的推测 | 可用性 |
|---|---|---|---|
| `marlin`（当前） | 混合精度专用融合 kernel | 可能按专家拆多次调用，kernel 数偏多 | ✅ |
| `triton` | Triton 写的通用 MoE kernel | 形状适应强，kernel 数随激活专家增长 | ✅ 待验证 |
| `batched_triton` | 多专家 GEMM 合并为一次批量调用 | **kernel 数更少**，可能对 decode 有利 | ✅ 待验证 |
| `deep_gemm` / `flashinfer_cutlass` / `flashinfer_trtllm` / `aiter` | 依赖 Hopper+ / CUTLASS / ROCm | — | ❌ sm86 不可用 |

切换方式（vLLM 0.30 的 `MoEBackend` Literal 中包含这些取值）：

```bash
--moe-backend marlin          # 当前 auto 选出的结果
--moe-backend triton
--moe-backend batched_triton

# 启动后核对实际生效的后端：
grep -i "MoE backend" "$run_dir/server.log"
```

## 一句话总结

> prefill 是算力-bound（正常）；decode 是 host-bound（每步几百个小 kernel，且 MoE 只能走兜底的 MARLIN）。
> 同时 TP 每层两次 all-reduce，因为"无 NVLink + 无 P2P + 跨 NUMA"全部退化为经主机内存的 SHM 通信，约 96 次/step 与 TPOT 同量级。
> → 两个有理论依据的优化方向：**换 MoE backend 减少 kernel 数**、**EP 替代 TP 减少通信次数**。

## 证据索引

| 事实 | 来源 |
|---|---|
| MarlinFP8ScaledMM / MoE backend = MARLIN / sm86 无原生 FP8 警告 | `runs/tp-nccl-graph-20261003T075549Z/server.log`（第 94、97、159 行附近） |
| FlashInfer AR 禁用、SymmMem 不支持 cap 8.6 | 同上（第 85–88 行附近） |
| NCCL 走 SHM、NVLS 不可用 | `artifacts/nccl-smoke.log`；`runs/*/server.log` |
| GPU0↔GPU1 = SYS、无 NVLink | `nvidia-smi topo -m`、`nvidia-smi nvlink --status` |
| P2P False、计算能力 (8,6)、NCCL 2.29.7 | `artifacts/runtime-check.txt` |
| TPOT / TTFT / 吞吐 | `runs/tp-nccl-graph-20261003T075549Z/bench/main/*.json`、`runs/tp-nccl-eager-20261003T081202Z/bench/main/*.json` |
| greedy 不可复现（含 eager 下仍分叉） | `runs/tp-nccl-baseline-20261003T072444Z/correctness-5x-32.json`、`runs/tp-nccl-eager-20261003T081202Z/correctness-5x-32.json` |

> 注：`runs/tp-nccl-graph-20261003T075549Z` 的 `run-meta.json` 里 `profile` 为 `tp-nccl`，目录名只是标签；它才是 TP 基线的 main 数据来源。
