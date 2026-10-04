"""Measure one TP0 gate/up GEMM with checkpoint weights and synthetic inputs.

This is a research fixture, not a serving benchmark or a model quality test.
Only explicitly running main() initializes CUDA. No installed vLLM file changes.
"""

import argparse
import importlib.metadata
import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_fp8_layer_for_marlin,
)
from vllm.scalar_type import scalar_types


def load_tp0_weights(model_dir, x_cpu, ids_cpu):
    """Load the first half of both intermediate projections for all experts."""
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    quantized, scales = [], []
    reference = torch.empty((8, 8, 768), dtype=torch.float32)

    with ExitStack() as stack:
        shards = {}

        def read(name):
            shard = weight_map[name]
            if shard not in shards:
                shards[shard] = stack.enter_context(
                    safe_open(model_dir / shard, framework="pt", device="cpu")
                )
            return shards[shard].get_tensor(name)

        for expert in range(128):
            prefix = f"model.layers.0.mlp.experts.{expert}"
            gate = read(f"{prefix}.gate_proj.weight")[:384].contiguous()
            up = read(f"{prefix}.up_proj.weight")[:384].contiguous()
            gate_scale = read(f"{prefix}.gate_proj.weight_scale_inv")[:3]
            up_scale = read(f"{prefix}.up_proj.weight_scale_inv")[:3]
            # Byte concatenation also works when CPU FP8 cat is unavailable.
            weight = torch.cat(
                (gate.view(torch.uint8), up.view(torch.uint8)), dim=0
            ).view(torch.float8_e4m3fn)
            scale = torch.cat((gate_scale, up_scale), dim=0).to(torch.bfloat16)

            token, slot = (ids_cpu == expert).nonzero(as_tuple=True)
            if token.numel():
                expanded = scale.float().repeat_interleave(128, 0)
                expanded = expanded.repeat_interleave(128, 1)
                # FP32 matmul on the same quantized weight values and BF16 scales.
                dequantized = weight.float() * expanded
                reference[token, slot] = x_cpu[token].float() @ dequantized.T

            layer = torch.nn.Module()
            layer.orig_dtype = torch.bfloat16
            layer.weight_block_size = [128, 128]
            layer.input_size_per_partition = 2048
            layer.output_size_per_partition = 768
            layer.weight = torch.nn.Parameter(weight.cuda(), requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(
                scale.cuda(), requires_grad=False
            )
            prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            quantized.append(layer.weight.detach())
            scales.append(layer.weight_scale_inv.detach())

    return torch.stack(quantized), torch.stack(scales), reference.reshape(64, 768)


def time_graph(call, repeats, rounds):
    """Batch graph nodes so Python replay overhead does not dominate one GEMM."""
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    nodes_per_graph = 20
    with torch.cuda.graph(graph):
        for _ in range(nodes_per_graph):
            call()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats / nodes_per_graph)
    return samples


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--thread-k", type=int, default=-1)
    parser.add_argument("--thread-n", type=int, default=-1)
    parser.add_argument("--blocks-per-sm", type=int, default=-1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("Source env.sh or provide --model-dir")
    if args.repeats < 1 or args.rounds < 1:
        parser.error("--repeats and --rounds must be positive")
    if (args.thread_k == -1) != (args.thread_n == -1):
        parser.error("Specify both --thread-k and --thread-n")
    if args.blocks_per_sm not in (-1, 1, 2, 3, 4):
        parser.error("--blocks-per-sm must be -1 or between 1 and 4")
    if os.environ.get("VLLM_MARLIN_INPUT_DTYPE"):
        parser.error("Unset VLLM_MARLIN_INPUT_DTYPE for this W8A16 experiment")

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    assert (config["hidden_size"], config["moe_intermediate_size"]) == (2048, 768)
    assert (config["num_experts"], config["num_experts_per_tok"]) == (128, 8)
    torch.cuda.set_device(0)
    if get_marlin_input_dtype() is not None:
        raise RuntimeError("This fixture requires BF16 activations")
    torch.manual_seed(args.seed)
    x_cpu = (torch.randn(8, 2048) / 10).to(torch.bfloat16)
    probabilities = torch.randn(8, 128).softmax(-1)
    routing_weights_cpu, ids_cpu = probabilities.topk(8)
    routing_weights_cpu /= routing_weights_cpu.sum(-1, keepdim=True)
    print("Loading layer 0 TP0 gate/up weights; synthetic input and routing.", flush=True)
    qweight, scales, reference = load_tp0_weights(model_dir, x_cpu, ids_cpu)
    x = x_cpu.cuda()
    routing_weights = routing_weights_cpu.cuda().contiguous()
    ids = ids_cpu.cuda().contiguous()
    sorted_ids, expert_ids, padded = moe_align_block_size(
        ids, 8, 128, ignore_invalid_experts=True
    )
    workspace = marlin_make_workspace_new(x.device, 4)
    output = torch.empty((64, 768), device=x.device, dtype=x.dtype)

    def call():
        return ops.moe_wna16_marlin_gemm(
            x, output, qweight, None, scales, None, None, None, workspace,
            sorted_ids, expert_ids, padded, routing_weights,
            moe_block_size=8, top_k=8, mul_topk_weights=False,
            b_q_type=scalar_types.float8_e4m3fn,
            size_m=8, size_n=768, size_k=2048,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
            thread_k=args.thread_k, thread_n=args.thread_n,
            blocks_per_sm=args.blocks_per_sm,
        )

    call()
    actual = output.float().cpu()
    if not torch.isfinite(actual).all():
        raise RuntimeError("Non-finite GEMM output; benchmark aborted")
    difference = actual - reference
    relative_l2 = (difference.norm() / reference.norm().clamp_min(1e-12)).item()
    # A screening threshold, not an end-to-end model quality acceptance rule.
    if relative_l2 > 0.02:
        raise RuntimeError(f"Reference relative L2 {relative_l2:.6f} exceeds 0.02")
    baseline_output = output.clone()
    latencies = time_graph(call, args.repeats, args.rounds)
    torch.cuda.synchronize()
    replay_max_difference = (output.float() - baseline_output.float()).abs().max().item()
    if not torch.isfinite(output).all():
        raise RuntimeError("Non-finite output after graph replay")

    result = {
        "scope": "layer0 TP0 W13 only; official weights, synthetic input/routing",
        "model_dir": str(model_dir),
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "shape": {"M": 8, "N": 768, "K": 2048, "E": 128, "topk": 8},
        "thread_k": args.thread_k,
        "thread_n": args.thread_n,
        "blocks_per_sm": args.blocks_per_sm,
        "active_experts": ids_cpu.unique().numel(),
        "routed_rows": 64,
        "padded_rows": padded.item(),
        "reference_relative_l2": relative_l2,
        "reference_max_abs_error": difference.abs().max().item(),
        "replay_max_abs_difference": replay_max_difference,
        "latency_us_per_call": latencies,
        "median_us": statistics.median(latencies),
        "timing": {"mode": "CUDA graph", "nodes_per_graph": 20,
                   "replays_per_round": args.repeats, "rounds": args.rounds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
