# Qwen3 MoE 双 RTX 3090 实验收尾报告

日期：2026-10-03；平台：AutoDL；本轮按用户要求停止新增 GPU 实验。

## 1. 本轮结论

1. 官方 Qwen3-30B-A3B-Instruct-2507-FP8 权重已经在同机双 3090 上运行，使用的是 **FP8 权重压缩 + BF16 激活的 Marlin 路径**。日志明确提示这台 GPU 不支持原生 FP8 计算。
2. 对固定 1024 输入 / 256 输出的随机负载，默认编译与 CUDA graph 路径下 TP=2 的输出吞吐优于本轮 EP=2。C=8 两轮均值分别为 **392.36 和 366.46 tok/s**，EP 低约 6.60%；EP 的 TTFT 更低，TPOT 更高。
3. eager 对照显著更慢，但 `--enforce-eager` 同时改变编译与 CUDA graph，不能把差异全部归因于 CUDA graph。
4. 全 trace 的 NCCL 占比约 63%～64%；分离纯 decode 后，C=8 的 **MoE Marlin 约占 kernel 耗时总和的 40%，NCCL 约占 26%**。优化 decode 需要关注 MoE 计算，不能直接套用全 trace 的通信占比。
5. 已把问题缩小到单卡 W13 探针：默认配置中位延迟 **88.05 μs**，数值检查与重复回放通过。尚未运行候选配置对照，因此没有证明性能提升，也没有可提交的性能优化 PR。

本轮价值是建立了“整机性能 → 阶段分析 → 对应源码 → 单算子测量”的证据链。生产验收、完整质量评估和新优化的验证仍未完成。

## 2. 实验对象与固定条件

| 项目 | 本轮实际条件 |
|---|---|
| 模型 | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` |
| revision | `5a5a776300a41aaa681dd7ff0106608ef2bc90db` |
| 权重 | 4 个 safetensors，共 31,175,618,584 bytes；下载时 SHA256 校验通过 |
| GPU | 2 × RTX 3090，24 GiB / 卡，计算能力 8.6 |
| 驱动 | 595.71.05；`nvidia-smi` 显示支持 CUDA 13.2，PyTorch 实际 runtime 为 13.0 |
| 软件 | vLLM 0.30.0，PyTorch 2.13.0+cu130，Transformers 5.18.0，NCCL 2.29.7 |
| 卡间路径 | 两方向 CUDA P2P 检查均 False；服务 NCCL 日志出现 `SHM/direct/direct` |
| Attention | TP=2；FlashAttention backend |
| 专家 | 48 个 MoE 层；每层 128 专家；每 token 选 8 个 |
| 基础服务参数 | DP=1，BF16 激活，KV dtype=auto，显存比例 0.85，最大上下文 8192 |
| 调度 | max-num-seqs=8，max-num-batched-tokens=2048，chunked prefill 开启，prefix caching 关闭 |
| 正式通信对照 | TP / EP / eager 都禁用 custom AllReduce，使用 NCCL |

模型是官方权重；本轮硬件、版本、参数组合是自己的实验基线。没有证据证明它对应某个上游认证的双 3090 性能合格线。

精确的启动命令、包列表与启动日志保存在每个 run 的 `command.sh`、`packages.txt`、`server.log`。权重身份及四分片哈希见 [model-manifest.json](../data/raw/2026-10-03/artifacts/model-manifest.json)；输入身份见快照 `workloads/hashes.json` 和正确性 JSON。

## 3. 功能与正确性：通过了什么

| 实验 | 用例 × 重复 | token 序列稳定 | 自动任务检查 | API 错误 |
|---|---:|---:|---:|---:|
| TP，初次检查 | 20 × 3 | 17/20 | 30/30 | 0 |
| TP，32-token 检查 | 20 × 5 | 19/20 | 50/50 | 0 |
| TP eager，32-token 检查 | 20 × 5 | 19/20 | 50/50 | 0 |
| EP，32-token 检查 | 20 × 5 | 18/20 | 50/50 | 0 |

32-token 检查中，TP 与 eager 的不稳定用例是 c20；EP 是 c19、c20。EP 与 TP 的首轮输出在 c19、c20 分别从 token 索引 16、10 开始不同。脚本的跨配置比较只比较每个用例的首轮，并未对所有重复两两交叉比较。

这些数据支持：API 能运行，所测自动任务未失败，部分开放用例存在 greedy 输出变化。**自动任务通过不等于逐 token 全部一致，也不等于完整模型质量通过。** 初次 3 轮检查和后续 32-token 检查不能当成同条件的“重复次数增加后改善”。

c11～c20 是人工检查项，归档中没有人工判定记录，因此保持“待人工检查”。没有 BF16 对照、logit margin 分析或 FP32 全模型参照，不能据此判定哪个分支正确，或断言存在内存损坏。

正式 graph 性能 run 是重启后的独立 run；TP 正确性材料来自之前的 TP baseline run。不能写成同一个进程同时通过了全部检查。

## 4. 正式性能结果

负载为随机文本，固定输入 1024、输出 256，seed=20261002，`ignore-eos`，无前缀缓存。每个配置 C=1/2/4/8 各测两轮，第二轮逆序；每轮请求数分别为 16/16/32/64，另有两条 warmup。

下表是**两轮指标的算术均值**。TTFT / TPOT 列分别是“两轮各自 P50 的均值”，不是把两轮请求合并后计算的 P50。TPOT 是请求内平均每输出 token 时间的分布指标，不等于全部 ITL 样本的 P50。

| 配置 | 并发 | 输出吞吐 tok/s | TTFT P50 均值 ms | TPOT P50 均值 ms |
|---|---:|---:|---:|---:|
| TP 编译 + graph | 1 | 130.20 | 293.99 | 6.56 |
| TP 编译 + graph | 2 | 200.55 | 429.66 | 8.30 |
| TP 编译 + graph | 4 | 290.10 | 932.85 | 10.23 |
| TP 编译 + graph | 8 | 392.36 | 1201.07 | 15.73 |
| EP 编译 + graph | 1 | 123.38 | 280.46 | 7.03 |
| EP 编译 + graph | 2 | 193.33 | 405.54 | 8.76 |
| EP 编译 + graph | 4 | 273.10 | 933.15 | 11.03 |
| EP 编译 + graph | 8 | 366.46 | 818.68 | 18.42 |
| TP eager | 1 | 17.07 | 302.12 | 57.67 |
| TP eager | 2 | 33.11 | 483.23 | 58.75 |
| TP eager | 4 | 64.91 | 557.71 | 59.35 |
| TP eager | 8 | 121.43 | 581.25 | 62.78 |

输出吞吐是测试总输出 token / 测试总时长，包含该负载下的 prefill 和排队影响，不能直接与其他人“纯 decode tok/s”的榜单比较。单请求速度也不能简单用并发总吞吐除以并发数来替代 TPOT。

24 个正式矩阵 JSON 共记录 768 次成功请求；加上 smoke 和 profiler 请求，共 **31 个 benchmark JSON，800 次成功、0 次失败**。这是请求完成情况，并非这些随机输出都做过语义检查。

正式矩阵的两轮吞吐极差 / 均值为 0.08%～1.83%；重复次数只有两轮，不能据此声称统计显著性。GPU 监控 CSV 已保存；graph / EP 时 GPU0 峰值温度分别 75℃ / 74℃，GPU1 均为 60℃，时钟未锁定。没有同一时刻对两个配置随机交替，也未完成优化 A→B→A。

原始逐轮、P95 与出处见 [bench-per-round.csv](../data/processed/2026-10-03/bench-per-round.csv)；聚合及两轮波动见 [bench-main-summary.csv](../data/processed/2026-10-03/bench-main-summary.csv)。不采用 benchmark 的 `max_concurrent_requests` 衍生字段解释调度容量；设置的并发来自命令及 metadata。

## 5. Profiler 与瓶颈判断

profiler 使用独立的 1024 输入 / 32 输出负载，C=1 两请求、C=8 八请求。它既缩短输出，又引入测量开销，数据单独分析，不并入上一节性能排名。以下采用 C=8 的 rank0 trace；rank1 原始 trace 同时归档。

### 5.1 全 trace

| 指标 | TP | EP |
|---|---:|---:|
| GPU kernel 数 | 39,272 | 39,272 |
| NCCL AllReduce kernel 数 | 3,589 | 3,589 |
| GPU busy 时间区间并集 ms | 2432.31 | 2516.62 |
| AllReduce 时间区间并集 ms | 1542.96 | 1621.79 |
| AllReduce / GPU busy | 63.44% | 64.44% |

分母是 GPU kernel 忙碌区间的并集，包含 prefill / mixed / decode，并非请求端到端耗时。不能说“端到端时间 64% 被通信占用”。trace 还有 profiler command-buffer 开销，尤其不能把重测量下的大 prefill 通信延迟直接解释成正常服务延迟。

### 5.2 只看纯 decode，batch=8

依据 GPU `execute_context_*(0)_generation_*(8)` 区间，选出各 26 个完整 decode step。每步覆盖 97 次 AllReduce；两配置各 2522 次，作为分类完整性核对。

| 分组 | TP kernel 耗时占比 | TP ms/step | EP kernel 耗时占比 | EP ms/step |
|---|---:|---:|---:|---:|
| MoE Marlin，gate/up + down | 40.19% | 4.400 | 40.33% | 4.626 |
| NCCL AllReduce | 25.64% | 2.807 | 25.70% | 2.948 |

这里占比的分母是所选区间的 kernel duration 总和；与上表的并集口径不同。GPU annotation 区间比 CPU launch 时间更适合划分阶段。分析脚本用 kernel 中点归属区间，并校验 decode collective 覆盖数；没有把覆盖不完整的 prefill 区间硬凑成一张百分比表。

97 次的代码解释是：embedding 归并 1 次，48 层各有 attention 输出归并、MoE 输出归并各 1 次，即 `1 + 48×2`。这是当前配置及 trace 的解释，不是任何 MoE 模型都必然如此。

**推断：** 当前纯 decode 的 MoE Marlin 是值得继续优化的热点。EP 没有减少这里的 AllReduce 次数，只改变了专家权重的分布和计算形状；只开启 EP 并不保证减少通信或提高吞吐。

可复算数据：[trace-full.csv](../data/processed/2026-10-03/trace-full.csv)、[trace-decode-batch8.csv](../data/processed/2026-10-03/trace-decode-batch8.csv)。

## 6. 两条候选方向的实际状态

### 6.1 自动通信路径：没有形成新对照

在 `tp-auto-profile-20261003T113524Z` 去掉 `--disable-custom-all-reduce` 后，启动日志仍提示 custom AllReduce 因 P2P 能力 / 测试失败而禁用，实际 backend 为 `['PYNCCL']`。这次只启动核对 backend，没有执行 C=1/C=8 性能矩阵。

因此这不是“custom 比 NCCL 更慢”的结果，也不是“本机所有通信优化都无效”。它说明这个开关没有在当前环境形成不同的实际执行路径。

### 6.2 单卡 W13：默认配置基线已完成

| 项目 | 实测 |
|---|---|
| 算子范围 | 第 0 层、TP rank0 专家 gate/up（W13），GPU0 单卡 |
| 数据 | 官方专家权重；合成 BF16 输入与合成路由，seed=7 |
| 形状 | M=8，N=768，K=2048，E=128，top-k=8 |
| 路由 | 45 个激活专家，64 条有效路由行，padding 后 360 行 |
| 运行参数 | thread_k=-1，thread_n=-1，blocks_per_sm=-1（自动选择） |
| 计时 | graph 内 20 次调用 × 每轮 100 次 replay × 5 轮 |
| 五轮 μs/调用 | 88.490 / 88.002 / 88.099 / 87.935 / 88.051 |
| 中位数 | 88.051 μs |
| 相对 FP32 反量化参照的 L2 误差 | 0.002278，约 0.228% |
| 最大绝对误差 | 0.0012064 |
| 同配置 replay 输出最大差 | 0 |

padding 有效行比例约 17.8%，这个比例不是 GPU 利用率。计时不包含权重加载、repack、路由生成 / 排序或 NCCL；反复使用同一权重与输入可能产生热缓存。它还没有包含 SiLU×up、W2、专家加权求和及完整模型质量检查。

脚本 0.02 的相对 L2 阈值仅用于筛选候选，不能替代模型精度验收。88 μs 也不能直接替换完整服务中的某层耗时。

证据：[marlin-w13-default-seed7.json](../data/raw/2026-10-03/artifacts/marlin-w13-default-seed7.json)；实际脚本在原始快照 `scripts/marlin_w13_probe.py`。

## 7. 问题、不足与下次改进

| 本轮不足 | 下次做法 |
|---|---|
| 安装和下载占用租机时间；最初 wheel 网络中断、环境没有 torch | 租机前核对 wheel / Python / CUDA 依赖；先准备可断点续传路径，缓存大文件 |
| tokenizer 缺失 / 加载路径导致 prepare_workloads 失败 | 权重以外，把 tokenizer、config、chat template 一并纳入离线加载检查与清单 |
| 原计划只测大配置，直到后段才分阶段定位 | 先取一份短 trace；区分 prefill、decode、CPU / launch、通信、算子，再决定下一条实验 |
| 看见 EP 开关容易默认存在 All-to-All、看见 FP8 容易默认原生 FP8 | 每次记录实际 backend、权重分布、dtype 与 trace kernel，不只记录 CLI 名称 |
| 把 greedy 漂移当成硬错误或把自动任务通过当成完整正确性 | 独立报告 API、任务、人工检查、序列重复性；有 margin / 高精度参照后再追根因 |
| eager 对照改变多个机制 | 要研究 graph 单独收益时固定编译条件，并核对实际执行路径 |
| profiler 整体百分比容易掩盖阶段热点 | 优先按 GPU 执行区间分类，保留分母定义和覆盖核对 |
| 单算子只测 seed=7、M=8，没有候选和 A→B→A | 先讨论假设；再小范围筛候选，覆盖多个 M、路由、种子及完整模型回归 |
| 温度 / 时钟与运行顺序可能影响结果 | 保留监控；短实验交替 A/B，有必要时固定电源与时钟条件 |
| 学习过程依赖 AI 发命令，尚无亲手 CPU MoE 参考实现的证据 | 自己复写小规模 router→top-k→专家→combine，逐项检查 tensor shape，再解释真实源码 |

没有可靠的实例账单和开关机时间记录，报告不估算最终费用。长上下文、真实请求集、超容量突发、持久稳定性等计划中的 suite 未在本轮执行，不填写“通过”。

## 8. 下次续接与贡献边界

下次从单卡探针继续，不重跑整套部署矩阵。先读 Python → 自定义算子 → C++ launch 选择代码，明确当前自动配置选了什么、为什么某候选可能更快，再执行一个候选。

一个可讨论的假设是：M 小、路由 padding 多时，默认 tile / blocks-per-SM 选择是否适合这个 GPU 和形状。它目前只是研究候选。要形成性能 PR，需要证明候选比当前默认稳定更快、数值可接受、其他形状无明显回退，并在拟贡献的上游 commit 上复验和检查重复工作。

目前没有上游 issue / PR 已提交，也没有“可 merge”的保证。

## 9. 本仓库中的数据与复算

`data/raw/2026-10-03/` 保存本轮 237 个原始材料文件，包括 8 个 run 的命令、JSON、日志、GPU 监控和两卡 trace。`third_party/vllm-0.30.0/` 是从实机安装包导出的 12 个 Python 链路源码文件，用于阅读，不参与 Python 导入或服务启动。

本地收尾时已经核验两份原始压缩包的远端与本地 SHA256 一致；此仓库提供解包后的文件，逐文件哈希见 `data/raw/2026-10-03/SHA256SUMS` 和源码目录的 `SHA256SUMS`。模型权重、虚拟环境和云实例系统进程清单没有作为仓库内容分发。

离线复算只需要 Python 标准库，无需 GPU、torch 或 vLLM：

```bash
python analysis/summarize_snapshot.py
```

输出写入 `data/processed/2026-10-03/`。JSON 和 CSV 的 `source` 指向仓库内原始文件，便于核对每个数字。
