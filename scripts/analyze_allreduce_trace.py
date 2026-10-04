#!/usr/bin/env python3

import gzip
import json
import math
from collections import Counter
from pathlib import Path


DTYPE_BYTES = {
    "c10::BFloat16": 2,
    "c10::Half": 2,
    "float": 4,
}


def percentile(values, ratio):
    values = sorted(values)
    if not values:
        return 0.0
    index = round((len(values) - 1) * ratio)
    return values[index]


def merge_intervals(intervals):
    """合并相互重叠的 [start, end] 时间区间。"""
    merged = []

    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


def interval_duration(intervals):
    return sum(end - start for start, end in intervals)


def newest_rank0_trace(pointer_file):
    run_dir = Path(pointer_file).read_text().strip()
    traces = sorted(
        Path(run_dir).glob(
            "traces/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz"
        )
    )

    if not traces:
        raise FileNotFoundError(f"No rank0 trace found under {run_dir}")

    return traces[-1]


def analyze(label, trace_path):
    with gzip.open(trace_path, "rt") as file:
        events = json.load(file)["traceEvents"]

    gpu_kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
    ]

    allreduce_kernels = [
        event
        for event in gpu_kernels
        if "ncclDevKernel_AllReduce" in event.get("name", "")
    ]

    gpu_intervals = [
        (event["ts"], event["ts"] + event["dur"])
        for event in gpu_kernels
    ]

    allreduce_intervals = [
        (event["ts"], event["ts"] + event["dur"])
        for event in allreduce_kernels
    ]

    gpu_sum_us = sum(event["dur"] for event in gpu_kernels)
    gpu_union_us = interval_duration(merge_intervals(gpu_intervals))

    ar_sum_us = sum(event["dur"] for event in allreduce_kernels)
    ar_union_us = interval_duration(
        merge_intervals(allreduce_intervals)
    )

    ar_durations = [
        event["dur"] for event in allreduce_kernels
    ]

    logical_shapes = Counter()

    for event in events:
        if (
            event.get("cat") != "cpu_op"
            or event.get("name") != "vllm::all_reduce"
        ):
            continue

        args = event.get("args", {})
        input_dims = args.get("Input Dims") or []
        input_types = args.get("Input type") or []

        if not input_dims or not input_dims[0]:
            continue

        shape = tuple(input_dims[0])
        dtype = input_types[0] if input_types else "unknown"
        logical_shapes[(shape, dtype)] += 1

    print(f"\n=== {label} ===")
    print(f"Trace: {trace_path}")
    print(f"GPU kernel count: {len(gpu_kernels)}")
    print(f"AllReduce kernel count: {len(allreduce_kernels)}")
    print(f"GPU kernel sum: {gpu_sum_us / 1000:.3f} ms")
    print(f"GPU kernel union: {gpu_union_us / 1000:.3f} ms")
    print(f"AllReduce sum: {ar_sum_us / 1000:.3f} ms")
    print(f"AllReduce union: {ar_union_us / 1000:.3f} ms")
    print(
        "AllReduce / GPU busy time: "
        f"{100 * ar_union_us / gpu_union_us:.2f}%"
    )
    print(
        "AllReduce kernel latency: "
        f"p50={percentile(ar_durations, 0.50):.3f} us, "
        f"p95={percentile(ar_durations, 0.95):.3f} us, "
        f"max={max(ar_durations):.3f} us"
    )

    print("Captured logical tensor shapes:")

    for (shape, dtype), count in logical_shapes.most_common():
        bytes_per_element = DTYPE_BYTES.get(dtype)
        if bytes_per_element is None:
            size_text = "unknown"
        else:
            size_bytes = math.prod(shape) * bytes_per_element
            size_text = f"{size_bytes / 1024 / 1024:.3f} MiB"

        print(
            f"  shape={shape}, dtype={dtype}, "
            f"count={count}, buffer={size_text}"
        )


def main():
    cases = [
        ("TP", newest_rank0_trace("tp-profile-run.txt")),
        ("EP", newest_rank0_trace("ep-profile-run.txt")),
    ]

    for label, trace_path in cases:
        analyze(label, trace_path)


if __name__ == "__main__":
    main()
