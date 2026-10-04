import os
from datetime import timedelta

import torch
import torch.distributed as dist

# torchrun 给每个进程分配 LOCAL_RANK：这里分别是 0 和 1。
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 两个进程加入同一个通信组，使用 NCCL 后端。
dist.init_process_group("nccl", timeout=timedelta(seconds=45))

try:
    rank = dist.get_rank()
    assert dist.get_world_size() == 2

    # rank 0 创建 [1,1,1,1]；rank 1 创建 [2,2,2,2]。
    x = torch.full(
        (4,), rank + 1,
        dtype=torch.float32,
        device=f"cuda:{local_rank}",
    )

    # 对应位置求和，并把结果交给两个进程。
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    assert (x == 3).all().item(), "求和结果错误"
    print(f"PASS rank={rank}, GPU={local_rank}, result={x.tolist()}", flush=True)
finally:
    dist.destroy_process_group()
