# 把本轮实验和代码对应起来

这里使用从租赁实例导出的**实际安装的 vLLM 0.30.0 源码**。Python 文件快照能解释运行路径；后续修改 C++ 时仍需取得对应版本的完整源码，不能默认本机另一个 `vllm-main` checkout 与安装包一致。

## 1. 先明确一个 token 在 MoE 层里经历什么

```text
hidden_states [M, 2048]
    ↓ router：为每行算 128 个专家分数
top-k：每行选择 8 个专家及其权重
    ↓ 按专家整理 token 行，并补齐 kernel 需要的块大小
每个被选专家：gate/up → SiLU(gate) × up → down
    ↓ 按路由权重加权求和
MoE output [M, 2048]
    ↓ 当前双卡配置需要合并卡上的局部结果
AllReduce 后进入下一层
```

专家是训练好的 FFN 参数，不是八个独立“对话机器人”。同一 token 的隐藏向量被送入所选专家，各专家做矩阵计算，再合并结果。

从 [Qwen3MoeSparseMoeBlock.forward](../third_party/vllm-0.30.0/model_executor/models/qwen3_moe.py#L215) 开始读。自己找出 router 输入 / 输出形状、experts 调用和最终返回形状。

## 2. TP 和 EP 真正改变了什么

| 本轮配置 | GPU0 / GPU1 的专家权重 | 算出来的局部输出 |
|---|---|---|
| TP=2 | 每张卡包含每层 128 个专家的矩阵分片 | 同一个专家的中间维分片计算结果 |
| Attention TP=2，专家 EP=2 | 每张卡包含每层 64 个完整专家 | 本卡所拥有专家的贡献 |

TP 中专家 intermediate=768 被分成两份 384。合并 gate/up 后，单卡 W13 输出维度是 768；EP 的完整专家 gate/up 输出维度是 1536。128 专家名称都在一张 TP 卡里，不等于 128 份完整专家权重都在里面。

本轮 EP 的 trace 仍有每步 97 次 AllReduce；不是仅打开 EP 就自动变成某种 All-to-All 加速路径。`--all2all-backend` 的配置字符串也不能证明某条 dispatch kernel 真正执行了。

## 3. 专家计算怎么落到 kernel

读 [Marlin MoE 实现](../third_party/vllm-0.30.0/model_executor/layers/fused_moe/experts/marlin_moe.py#L60)。按以下顺序找调用：

1. top-k 结果如何变成 `sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`。
2. 第一次 `ops.moe_wna16_marlin_gemm` 计算 gate/up（W13）。
3. 激活函数将 gate 与 up 合并。
4. 第二次 GEMM 计算 down（W2）。
5. router weight 在什么位置应用，最终如何合并。

然后读 [自定义算子封装](../third_party/vllm-0.30.0/_custom_ops.py#L2509)，找到 `thread_k`、`thread_n`、`blocks_per_sm` 怎样传进 native 算子。

这些参数控制 kernel 计算与调度方式，不改变模型的专家总数。传 `-1` 代表让实现选择默认值；不是 thread 数等于 -1。

## 4. 为什么 3090 上 FP8 仍然值得研究

服务日志明确选择了 Marlin FP8，并提示没有原生 FP8 计算支持。先读 [FP8 Marlin 权重准备逻辑](../third_party/vllm-0.30.0/model_executor/layers/quantization/utils/marlin_utils_fp8.py#L110)。

本轮保存低比特权重，计算时使用 weight-only 路径；激活是 BF16。权重占用小和硬件支持原生 FP8 矩阵乘法是两件需要分别验证的事实。

## 5. 通信调用链

本轮三类主要入口是 embedding、Attention 的 RowParallelLinear 输出、MoE 的最终输出。

```text
模型层的局部输出
  → tensor_model_parallel_all_reduce
  → TP group 的 all_reduce
  → CudaCommunicator 的 backend 分派
  → 当前实际 PYNCCL
  → trace 里的 NCCL AllReduce kernel
```

可以按顺序打开：

- [MoE 最终归并](../third_party/vllm-0.30.0/model_executor/layers/fused_moe/runner/moe_runner.py#L480)。
- [tensor_model_parallel_all_reduce](../third_party/vllm-0.30.0/distributed/communication_op.py#L12)。
- [group all_reduce](../third_party/vllm-0.30.0/distributed/parallel_state.py#L721)。
- [CudaCommunicator](../third_party/vllm-0.30.0/distributed/device_communicators/cuda_communicator.py#L327)。

把每步 97 次的结构与代码对应：embedding 1，48 层的 attention 48，48 层的 MoE 48。这能解释次数，没有证明每一处都可以删掉。删掉必要的归并会改变计算结果。

## 6. 单卡探针在回答哪个问题

实际运行脚本：[marlin_w13_probe.py](../data/raw/2026-10-03/scripts/marlin_w13_probe.py)。

它拿第 0 层官方权重，切出 TP0 的 gate/up，准备 Marlin 所需的权重布局，生成 8 行合成输入和 top-8 路由，只计 W13。它回答“固定数据下，这个 GEMM 的默认配置多快、数值是否接近参照”。

它没有运行完整专家、整层网络或 NCCL。固定输入多次 replay 输出一样，只能证明这个探针条件下的重复性，不能推出整个生成服务必然确定。

路由例子：8 行 × 8 专家 = 64 次 token-expert 计算；这次共触及 45 专家，每专家至少对齐到 8 行，总 padded rows=360。多数专家拿到的有效行很少，可以提出调度优化假设；但必须实际测量才知道 padding 是否能减少、减少后是否更快。

## 7. 下次由你亲手完成的三个小练习

1. 在 CPU 用小维度写 router → top-k → 两个专家矩阵 → 加权求和，解释每一步的 shape。
2. 对照探针，逐行标出“构造输入”“权重准备”“数值参照”“计时”四部分，解释为什么准备时间不混入 kernel 时间。
3. 读对应版本 C++ 的默认 launch selector，先写下一个可证伪的预测，再讨论是否租卡验证。不要先搜索一长串开关挨个试。

这三项尚未算完成；本轮归档用于后续学习，不把 AI 写出的脚本直接当成你已经掌握的工程能力。
