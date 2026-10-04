"""Rebuild tables from the archived experiment; no GPU or vLLM import needed."""

import csv
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "data/raw/2026-10-03"
OUT = LAB / "data/processed/2026-10-03"
METRICS = ("output_throughput", "p50_ttft_ms", "p95_ttft_ms",
           "p50_tpot_ms", "p95_tpot_ms", "p50_e2el_ms")


def save_csv(name, rows):
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def union_us(intervals):
    total = 0.0
    right = None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
            right = end
        elif end > right:
            total += end - right
            right = end
    return total


def kernel_group(name):
    if "ncclDevKernel_AllReduce" in name:
        return "NCCL AllReduce"
    if "marlin_moe_wna16::Marlin" in name:
        return "MoE Marlin"
    if "flash::" in name:
        return "FlashAttention"
    if "marlin::Marlin" in name:
        return "Dense Marlin"
    if name.startswith("triton_"):
        return "Triton"
    if "topkGating" in name:
        return "Router top-k"
    if "moe_align_block_size" in name or "count_and_sort_expert_tokens" in name:
        return "Routing alignment"
    return "Other"


def analyze_trace(label, path):
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    ar = [e for e in kernels if kernel_group(e["name"]) == "NCCL AllReduce"]
    gpu_busy = union_us((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    ar_busy = union_us((e["ts"], e["ts"] + e["dur"]) for e in ar)
    full = dict(profile=label, source=path.relative_to(LAB).as_posix(),
                kernel_count=len(kernels), allreduce_count=len(ar),
                kernel_sum_ms=sum(e["dur"] for e in kernels) / 1000,
                gpu_busy_union_ms=gpu_busy / 1000,
                allreduce_union_ms=ar_busy / 1000,
                allreduce_pct_gpu_busy=100 * ar_busy / gpu_busy)
    # Select GPU execution annotations, not asynchronous CPU launch timestamps.
    pattern = re.compile(r"execute_context_\d+\((\d+)\)_generation_\d+\((\d+)\)")
    streams = defaultdict(list)
    for e in events:
        m = pattern.fullmatch(e.get("name", ""))
        if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X" and m:
            streams[(e["pid"], e["tid"])].append((e, int(m[1]), int(m[2])))
    if not streams:
        raise ValueError(f"No GPU execution annotations: {path}")
    forward = max(streams.values(), key=len)
    decode8 = [(e["ts"], e["ts"] + e["dur"]) for e, ctx, gen in forward if ctx == 0 and gen == 8]
    if not decode8:
        raise ValueError(f"No pure C=8 decode steps: {path}")
    # Midpoint assignment is adequate only for fully covered decode steps.
    # Check the expected collectives so incomplete phase coverage is not silent.
    selected = [e for e in kernels if any(a <= e["ts"] + e["dur"] / 2 < b for a, b in decode8)]
    groups = defaultdict(list)
    for e in selected:
        groups[kernel_group(e["name"])].append(e)
    if len(groups["NCCL AllReduce"]) != 97 * len(decode8):
        raise ValueError(f"Incomplete decode collective coverage: {path}")
    total = sum(e["dur"] for e in selected)
    phase = []
    for name, es in sorted(groups.items(), key=lambda p: -sum(e["dur"] for e in p[1])):
        dur = sum(e["dur"] for e in es)
        phase.append(dict(profile=label, phase="pure_decode_batch8", steps=len(decode8),
                          group=name, kernel_count=len(es), kernel_sum_ms=dur / 1000,
                          pct_kernel_sum=100 * dur / total,
                          ms_per_step=dur / 1000 / len(decode8),
                          source=path.relative_to(LAB).as_posix()))
    full.update(forward_annotations=len(forward), decode_batch8_steps=len(decode8))
    return full, phase


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, grouped, correctness = [], defaultdict(list), []
    for p in sorted((RAW / "runs").glob("*/bench/*/*.json")):
        d = json.loads(p.read_text())
        if "completed" not in d:
            continue
        assert d["failed"] == 0 and d["completed"] == d["num_prompts"], p
        row = dict(run=p.parents[2].name, profile=d["profile"], suite=d["suite"],
                   concurrency=int(d["concurrency"]), round=int(d["round"]),
                   completed=d["completed"], failed=d["failed"], duration_s=d["duration"],
                   input_tokens=d["total_input_tokens"], output_tokens=d["total_output_tokens"],
                   **{k: d[k] for k in METRICS}, source=p.relative_to(LAB).as_posix())
        rows.append(row)
        if row["suite"] == "main":
            grouped[(row["run"], row["profile"], row["concurrency"])].append(row)
    aggregates = []
    for (run, profile, concurrency), rs in sorted(grouped.items()):
        assert len(rs) == 2 and {r["round"] for r in rs} == {1, 2}
        means = {k + "_round_mean": statistics.mean(r[k] for r in rs) for k in METRICS}
        throughputs = [r["output_throughput"] for r in rs]
        aggregates.append(dict(run=run, profile=profile, concurrency=concurrency, rounds=len(rs),
                               completed=sum(r["completed"] for r in rs), failed=sum(r["failed"] for r in rs),
                               **means, throughput_round_spread_pct=100 * (max(throughputs) - min(throughputs)) / statistics.mean(throughputs)))
    for p in sorted((RAW / "runs").glob("*/correctness*.json")):
        d = json.loads(p.read_text())
        groups = defaultdict(list)
        for row in d["rows"]:
            groups[row["id"]].append(row)
        unstable = [name for name, rs in groups.items() if len({tuple(r["token_ids"]) for r in rs}) > 1]
        summary = d["summary"]
        assert len(groups) - len(unstable) == summary["stable_cases"]
        correctness.append(dict(run=p.parent.name, file=p.name, **summary,
                                unstable_case_ids=unstable, manual_review_status="not recorded",
                                source=p.relative_to(LAB).as_posix()))
    full, phases = [], []
    for label, run in [("TP", "tp-nccl-profile-20261003T103620Z"), ("EP", "ep-nccl-profile-20261003T104909Z")]:
        path = sorted((RAW / "runs" / run / "traces").glob("dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz"))[-1]
        f, ps = analyze_trace(label, path)
        full.append(f)
        phases.extend(ps)
    save_csv("bench-per-round.csv", rows)
    save_csv("bench-main-summary.csv", aggregates)
    save_csv("trace-full.csv", full)
    save_csv("trace-decode-batch8.csv", phases)
    result = dict(aggregation="arithmetic mean of the two round metrics; percentile values are not pooled percentiles",
                  benchmarks=aggregates, correctness=correctness, trace_full=full,
                  trace_decode_batch8=phases, marlin_baseline=json.loads((RAW / "artifacts/marlin-w13-default-seed7.json").read_text()))
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(rows)} benchmark rows, {len(aggregates)} matched summaries, {len(correctness)} correctness records")
    for r in aggregates:
        print(r["profile"], "C=" + str(r["concurrency"]),
              f"{r['output_throughput_round_mean']:.3f} tok/s",
              f"TTFT {r['p50_ttft_ms_round_mean']:.3f} ms",
              f"TPOT {r['p50_tpot_ms_round_mean']:.3f} ms")
    print("Profiler:", json.dumps(full, ensure_ascii=False))


if __name__ == "__main__":
    main()
