"""S-SPPO similarity of each selected pair: sim_score = max(0, cos(phi(y_w), phi(y_l)))."""
import os
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import sys

import patch_transformers_torchao  # noqa: F401
import numpy as np
import pandas as pd

from prompt_utils import write_parquet_atomic


def _extract_assistant(msgs):
    if isinstance(msgs, str):
        return msgs
    # pandas returns a parquet list<struct> column as a numpy array, not a list.
    if isinstance(msgs, np.ndarray):
        msgs = msgs.tolist()
    if isinstance(msgs, (list, tuple)) and len(msgs) >= 2 and isinstance(msgs[-1], dict):
        return msgs[-1].get("content", "") or ""
    return ""


def encode(texts, model_name, batch_size, device):
    from sentence_transformers import SentenceTransformer

    print(f"[compute_sim] loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name, device=device)
    # normalize_embeddings=True so a dot product is the cosine similarity.
    return encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="<score_dir>/train.parquet from compute_prob.py")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default=None, help="e.g. cuda / cpu / cuda:0 (default: sentence-transformers auto)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[compute_sim] parquet not found: {args.input}")
    df = pd.read_parquet(args.input)
    n = len(df)
    chosen = [_extract_assistant(m) for m in df["chosen"].tolist()]
    rejected = [_extract_assistant(m) for m in df["rejected"].tolist()]

    n_empty = sum(1 for t in chosen + rejected if not t.strip())
    if n_empty > 0.01 * 2 * n:
        sys.exit(f"[compute_sim] {n_empty} of {2 * n} responses extracted as empty; "
                 "the chat column layout is not what this script expects.")
    if n_empty:
        print(f"[compute_sim] warning: {n_empty} of {2 * n} responses are empty (policy emitted EOS).")

    emb = encode(chosen + rejected, args.model, args.batch_size, args.device)
    sim = np.clip((emb[:n] * emb[n:]).sum(axis=1), 0.0, None).astype(np.float32)

    df = df.drop(columns=[c for c in ("sim_score_chosen", "sim_score_rejected") if c in df.columns])
    df["sim_score"] = sim
    print(f"[compute_sim] sim_score: mean={sim.mean():.4f} std={sim.std():.4f} min={sim.min():.4f} max={sim.max():.4f}")
    write_parquet_atomic(df, args.input)
    print(f"[compute_sim] wrote {n} rows -> {args.input}")


if __name__ == "__main__":
    main()
