"""Build the SPPO preference dataset from the K samples and their PairRM scores."""
import os
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import json

import numpy as np
import pandas as pd

from prompt_utils import load_prompt_dataset, num_shards, write_parquet_atomic


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--rank_dir", type=str, required=True)
    parser.add_argument("--score_dir", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="data/Ultrachat50k/train.jsonl")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, required=True)
    return parser.parse_args()


def load_shards(args, n):
    responses = [[] for _ in range(args.pairs)]
    scores = []
    for sid in range(num_shards(n, args.frac_len)):
        for p in range(args.pairs):
            with open(os.path.join(args.gen_dir, f"responses_{sid}_{p}.json")) as f:
                responses[p].extend(json.load(f))
        scores.append(np.load(os.path.join(args.rank_dir, f"{sid}.npy")))
    scores = np.concatenate(scores, axis=0)
    if scores.shape != (n, args.pairs) or any(len(r) != n for r in responses):
        raise SystemExit(
            f"[compute_prob] expected {n} prompts x {args.pairs} samples; got scores {scores.shape}, "
            f"responses {[len(r) for r in responses]}"
        )
    return responses, scores


def main():
    args = parse_arguments()
    data = load_prompt_dataset(args.prompts)
    n = len(data)
    prompts = data["prompt"]
    responses, scores = load_shards(args, n)

    # probs[r, i, j] = P(y_i > y_j) = sigmoid(s_i - s_j)
    probs = 1.0 / (1.0 + np.exp(scores[:, None, :] - scores[:, :, None]))
    p_hat = probs.mean(axis=2)
    win = scores.argmax(axis=1)
    lose = scores.argmin(axis=1)
    rows = np.arange(n)

    def chat(r, p):
        return [{"content": prompts[r], "role": "user"}, {"content": responses[p][r], "role": "assistant"}]

    generated = data.to_pandas()
    for p in range(args.pairs):
        generated[f"generate_{p}"] = [chat(r, p) for r in range(n)]
    generated["probability"] = [m.tolist() for m in probs]
    generated["rm_scores"] = [s.tolist() for s in scores]
    write_parquet_atomic(generated, os.path.join(args.gen_dir, "train.parquet"))

    scored = pd.DataFrame({
        "chosen": [chat(r, win[r]) for r in range(n)],
        "rejected": [chat(r, lose[r]) for r in range(n)],
        "chosen_probs": probs[rows, win, lose],
        "chosen_probs_win": p_hat[rows, win],
        "chosen_probs_lose": p_hat[rows, lose],
    })
    os.makedirs(args.score_dir, exist_ok=True)
    stale_test = os.path.join(args.score_dir, "test.parquet")
    if os.path.exists(stale_test):
        os.remove(stale_test)
    write_parquet_atomic(scored, os.path.join(args.score_dir, "train.parquet"))
    print(f"[compute_prob] wrote {n} rows -> {args.score_dir}/train.parquet")


if __name__ == "__main__":
    main()
