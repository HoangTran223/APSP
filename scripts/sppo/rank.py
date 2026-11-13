"""Score the K responses of one shard with PairRM; writes <rank_dir>/<data_frac>.npy of shape (n, K)."""
import os
os.environ.setdefault("WANDB_MODE", "offline")

import patch_transformers_torchao  # noqa: F401
import argparse
import json

import llm_blender
import numpy as np
from transformers import AutoTokenizer

from prompt_utils import RANKER, apply_template, load_prompt_dataset, shard, tmp_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="policy whose chat template formats the prompt")
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--rank_dir", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="data/Ultrachat50k/train.jsonl")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--data_frac", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    out = os.path.join(args.rank_dir, f"{args.data_frac}.npy")
    if os.path.exists(out):
        print(f"[rank] {out} exists")
        return

    data = load_prompt_dataset(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = shard([apply_template(text, tokenizer) for text in data["prompt"]], args.frac_len, args.data_frac)

    per_sample = []
    for p in range(args.pairs):
        with open(os.path.join(args.gen_dir, f"responses_{args.data_frac}_{p}.json")) as f:
            per_sample.append(json.load(f))
    if any(len(responses) != len(prompts) for responses in per_sample):
        raise SystemExit(
            f"[rank] shard {args.data_frac}: {len(prompts)} prompts but response counts "
            f"{[len(r) for r in per_sample]}"
        )
    candidates = [list(row) for row in zip(*per_sample)]

    blender = llm_blender.Blender()
    blender.loadranker(RANKER)
    scores = np.asarray(blender.rank(prompts, candidates, return_scores=True, batch_size=1))

    os.makedirs(args.rank_dir, exist_ok=True)
    tmp = tmp_path(out)
    with open(tmp, "wb") as f:
        np.save(f, scores)
    os.replace(tmp, out)
    print(f"[rank] wrote {out} {scores.shape}")


if __name__ == "__main__":
    main()
