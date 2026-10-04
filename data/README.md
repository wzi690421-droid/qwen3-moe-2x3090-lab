# 数据目录

`raw/2026-10-03/` 是本轮从实例备份的原始材料，文件内容保留不变；其中绝对路径反映当时实例的目录。`processed/2026-10-03/` 由本仓库分析脚本生成，source 指向可浏览的仓库路径。

原始材料共 237 个文件，包含 8 个 run、31 个 benchmark JSON、4 份正确性 JSON、两卡 profiler trace 和监控。原始 README 是历史计划，以根 README 和 docs 的实测状态为阅读入口。

## 来源与哈希

本机原始 evidence 压缩包 SHA256：

`322f63bd00bf74cdf09ea42be084325255d9f59aad1f1b3efb4142d34a06d233`

安装源码与收尾状态压缩包 SHA256：

`09c24bd5d123029e6a097ad9526190c21865fba4bbfad9b8616821acc22d4490`

本仓库提供解包的实验原始文件和 12 个安装源码文件，没有重复存储两份压缩包，也没有分发实例系统进程清单。逐文件 SHA256 位于 raw 和 third_party 子目录。无需 GPU 即可重新汇总：

```bash
python analysis/summarize_snapshot.py
```

微基准记录属于固定合成输入 / 路由的测量；性能图中的两轮区间只是观测极差，不是统计置信区间。
