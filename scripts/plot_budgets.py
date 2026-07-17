#!/usr/bin/env python3
"""Plot mean layer-head EntroKV budgets for one or more tasks."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_budget(path):
    by_layer = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_layer.setdefault(int(record["layer"]), []).append(record["history_budget"])
    if not by_layer:
        raise ValueError(f"no budget records in {path}")
    return np.asarray([np.mean(by_layer[layer], axis=0) for layer in sorted(by_layer)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="TASK=budget.jsonl")
    parser.add_argument("--output", default="outputs/budget_heatmaps.png")
    args = parser.parse_args()

    items = []
    for item in args.inputs:
        task, path = item.split("=", 1)
        items.append((task, load_budget(path)))

    fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 8), squeeze=False)
    for axis, (task, matrix) in zip(axes[0], items):
        sns.heatmap(matrix, cmap="viridis", ax=axis, cbar_kws={"label": "Mean history KV budget"})
        axis.set_title(task)
        axis.set_xlabel("KV head")
        axis.set_ylabel("Layer")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
