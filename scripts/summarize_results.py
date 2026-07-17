#!/usr/bin/env python3
"""Summarize efficiency fields from LongBench prediction JSONL files."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", nargs="+")
    parser.add_argument("--output", default="outputs/efficiency_summary.json")
    args = parser.parse_args()

    summary = {}
    for filename in args.prediction_files:
        rows = [json.loads(line) for line in open(filename, encoding="utf-8")]
        summary[str(filename)] = {
            "samples": len(rows),
            "mean_latency_s": float(np.mean([row["latency_s"] for row in rows])),
            "mean_tokens_per_second": float(np.mean([row["tokens_per_second"] for row in rows])),
            "max_peak_memory_mb": float(np.max([row["peak_memory_mb"] for row in rows])),
            "mean_input_tokens": float(np.mean([row["input_tokens"] for row in rows])),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
