"""Export official benchmark result JSON files to one CSV; do not invent missing values."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for meta in sorted(args.runs.glob("*/run-meta.json")):
        data = json.loads(meta.read_text())
        for path in sorted((meta.parent / "bench").glob("*/*.json")):
            result = json.loads(path.read_text())
            if "completed" not in result:
                continue
            rows.append({"run": meta.parent.name, "profile": data["profile"],
                         "suite": path.parent.name, "file": path.name,
                         "duration_s": result.get("duration"),
                         "completed": result.get("completed"), "failed": result.get("failed"),
                         "output_tokens": result.get("total_output_tokens"),
                         "output_tok_s": result.get("output_throughput"),
                         "p50_ttft_ms": result.get("p50_ttft_ms"),
                         "p95_ttft_ms": result.get("p95_ttft_ms"),
                         "p50_tpot_ms": result.get("p50_tpot_ms"),
                         "p95_tpot_ms": result.get("p95_tpot_ms"),
                         "p50_e2el_ms": result.get("p50_e2el_ms"),
                         "source": str(path.resolve())})
    if not rows:
        raise SystemExit("No completed benchmark files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} result rows to {args.out}")


if __name__ == "__main__":
    main()
