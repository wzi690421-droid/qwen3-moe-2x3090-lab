"""Small two-rank NCCL AllReduce probe. This is not a raw NVLink bandwidth test."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    out = Path(args.out)
    if out.exists():
        parser.error("Use a new output filename")
    import torch
    import torch.distributed as dist
    assert int(os.environ["WORLD_SIZE"]) == 2
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        rows = []
        for size in [4096, 65536, 1048576, 8388608]:
            x = torch.empty(size // 4, device=f"cuda:{local_rank}", dtype=torch.float32)
            for _ in range(10):
                x.fill_(local_rank + 1)
                dist.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            events = []
            for _ in range(args.iterations):
                # Reset input; otherwise repeated in-place sums grow exponentially.
                x.fill_(local_rank + 1)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(x)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            assert torch.all(x == 3).item(), "AllReduce sum failed"
            times = [start.elapsed_time(end) for start, end in events]
            per_rank = [None, None]
            dist.all_gather_object(per_rank, times)
            slow_rank_times = [max(t) for t in zip(*per_rank)]
            median_ms = statistics.median(slow_rank_times)
            rows.append({"payload_bytes_per_rank": size, "dtype": "float32",
                         "median_ms": median_ms,
                         "payload_GB_per_s": size / (median_ms / 1000) / 1e9,
                         "per_rank_ms": per_rank, "correct_sum": True})
        if dist.get_rank() == 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
                    "iterations": args.iterations, "rows": rows,
                    "scope": "NCCL AllReduce payload throughput, not physical link bandwidth"}
            out.write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps(data, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
