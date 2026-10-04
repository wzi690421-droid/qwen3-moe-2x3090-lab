"""Run sequential greedy smoke tests; separately report task checks and token repeatability."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def check_task(case, text):
    if case["check"] == "manual":
        return None
    if case["check"] == "exact":
        return text.strip() == case["expected"]
    if case["check"] == "json":
        try:
            value = json.loads(text)
            return value == case["expected"] and type(value) is type(case["expected"])
        except (ValueError, TypeError):
            return False
    raise ValueError(f"Unknown check: {case['check']}")


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", help="Previous saved check JSON, for cross-profile comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    root = Path(os.environ["LAB_DIR"])
    out = Path(args.out)
    if out.exists():
        parser.error("Use a new output filename; do not overwrite evidence")
    out.parent.mkdir(parents=True, exist_ok=True)
    input_path = root / "workloads/prepared-cases.json"
    cases = json.loads(input_path.read_text())
    result = {"model_revision": os.environ["MODEL_REVISION"],
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "rows": [], "summary": {}}

    for rep in range(args.repeats):
        for case in cases:
            body = {"model": os.environ["SERVED_MODEL"], "prompt": case["prompt_token_ids"],
                    "temperature": 0, "top_p": 1, "seed": 20261002,
                    "max_tokens": args.max_tokens, "stream": False,
                    "return_token_ids": True, "logprobs": 2}
            row = {"id": case["id"], "repeat": rep + 1, "request": body,
                   "prompt": case["prompt"], "manual_check": case["check"] == "manual"}
            start = time.perf_counter()
            try:
                request = urllib.request.Request(args.base_url + "/v1/completions",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                tokens = choice.get("token_ids")
                if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
                    raise ValueError("Missing generated token_ids; do not substitute retokenized text")
                text = choice["text"]
                if not text.strip():
                    raise ValueError("Empty completion")
                row.update(text=text, token_ids=tokens, finish_reason=choice.get("finish_reason"),
                           task_pass=check_task(case, text), logprobs=choice.get("logprobs"),
                           usage=data.get("usage"), error=None)
            except (urllib.error.URLError, ValueError, KeyError, TimeoutError, OSError) as exc:
                row["error"] = str(exc)
            row["elapsed_seconds"] = time.perf_counter() - start
            result["rows"].append(row)
            # Preserve every completed request if the next request or the server fails.
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"repeat={rep+1} {case['id']} error={row.get('error')} task={row.get('task_pass')}", flush=True)
            if row.get("error"):
                result["summary"] = {"aborted": True, "reason": row["error"]}
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                raise SystemExit(2)

    first = {r["id"]: r for r in result["rows"] if r["repeat"] == 1}
    stable = sum(all(r["token_ids"] == first[c["id"]]["token_ids"]
                     for r in result["rows"] if r["id"] == c["id"]) for c in cases)
    hard = [r for r in result["rows"] if r.get("task_pass") is not None]
    summary = {"api_errors": 0, "cases": len(cases), "repeats": args.repeats,
               "stable_cases": stable, "automatic_task_checks": len(hard),
               "automatic_task_failures": sum(not r["task_pass"] for r in hard),
               "manual_case_ids": [c["id"] for c in cases if c["check"] == "manual"]}
    if args.baseline:
        old = json.loads(Path(args.baseline).read_text())
        if old["input_sha256"] != result["input_sha256"] or old["model_revision"] != result["model_revision"]:
            raise ValueError("Baseline model revision / prompt inputs differ")
        old_first = {r["id"]: r for r in old["rows"] if r["repeat"] == 1}
        differences = []
        for case_id, row in first.items():
            prior = old_first[case_id]
            if prior["request"] != row["request"]:
                raise ValueError("Baseline request settings differ")
            position = first_difference(prior["token_ids"], row["token_ids"])
            if position is not None:
                differences.append({"id": case_id, "first_different_token_index": position})
        summary["cross_profile_differences"] = differences
    result["summary"] = summary
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Exit 2 is a diagnostic requiring review, not a proof that the model is defective.
    if summary["automatic_task_failures"] or stable != len(cases):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
