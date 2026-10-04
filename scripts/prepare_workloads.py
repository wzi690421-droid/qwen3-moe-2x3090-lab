"""Render fixed correctness prompts and a small, distinct real-text benchmark corpus."""
import hashlib
import json
import os
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    root = Path(os.environ["LAB_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"], local_files_only=True)
    cases = json.loads((root / "workloads/cases.json").read_text())
    for case in cases:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # Transformers 5.x may return BatchEncoding instead of list[int].
        if hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        case["prompt_token_ids"] = [int(token_id) for token_id in token_ids]
    prepared = root / "workloads/prepared-cases.json"
    prepared.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    # Natural text, multiple domains; each request has different source material.
    # This is a learning corpus, not a production workload or an accuracy benchmark.
    tasks = [
        "请总结这份故障记录，区分观测事实和待验证假设，并给出下一项实验。",
        "请为这个服务设计一个 Python 指标汇总函数，说明输入、输出和空数据处理。",
        "请把这份记录翻译成英文，保留数字、单位和不确定性。",
        "请写一份简短的工程评审，说明性能、延迟与费用的取舍。",
    ]
    lines = []
    for i in range(32):
        material = [f"记录编号 {i}，客户端使用串行与并发请求；每条记录的版本和请求长度均单独保存。"]
        for j in range(12):
            material.append(f"批次 {i}-{j}：输入 {512 + j * 64} tokens，输出上限 256 tokens，"
                            f"并发 {1 + (i+j) % 8}，聚合吞吐 {90 + i + j} token/s，"
                            "无 HTTP 错误；硬件时钟、温度和请求排队时间仍需核对。")
        lines.append(json.dumps({"prompt": tasks[i % len(tasks)] + "\n" + "\n".join(material),
                                 "output_tokens": 256}, ensure_ascii=False))
    real = root / "workloads/real.jsonl"
    real.write_text("\n".join(lines) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [prepared, real]}
    (root / "workloads/hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Prepared {len(cases)} correctness cases and {len(lines)} real-text requests.")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
