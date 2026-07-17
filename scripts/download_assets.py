"""Download the gated model and selected LongBench tasks on a CUDA machine.

Usage:
  HF_TOKEN=hf_... python scripts/download_assets.py --data-only
  HF_TOKEN=hf_... python scripts/download_assets.py --model-only
"""
import argparse
import os
from pathlib import Path


TASKS = ("narrativeqa", "hotpotqa", "qasper", "passage_retrieval_en")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    if args.data_only and args.model_only:
        parser.error("choose at most one of --data-only and --model-only")

    token = os.environ.get("HF_TOKEN")
    Path("data/longbench").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if not args.model_only:
        from datasets import load_dataset

        for task in TASKS:
            print(f"Caching LongBench task: {task}")
            load_dataset("zai-org/LongBench", task, split="test", token=token)

    if not args.data_only:
        if not token:
            raise SystemExit("HF_TOKEN is required for the gated Llama model")
        from huggingface_hub import snapshot_download

        print(f"Downloading model: {args.model}")
        snapshot_download(
            repo_id=args.model,
            token=token,
            local_dir=Path("models") / args.model.rsplit("/", 1)[-1],
        )


if __name__ == "__main__":
    main()
