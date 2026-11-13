#!/usr/bin/env python3
# UCLA-AGI/SPIN_iter0 parquet -> UltraChat50k {prompt, ground_truth} jsonl.

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def messages_to_record(messages):
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None
    parts = []
    for m in messages[:-1]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant" if role == "assistant" else role.capitalize()
        parts.append(f"{label}: {content}")
    if not parts:
        return None
    gt = (messages[-1].get("content") or "").strip()
    if not gt:
        return None
    return {"prompt": "\n\n".join(parts), "ground_truth": gt}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--prompts_out", type=Path, required=True)
    p.add_argument("--max_rows", type=int, default=-1)
    args = p.parse_args()

    table = pq.read_table(args.parquet, columns=["real", "generated"])
    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_mismatch = 0
    with args.prompts_out.open("w", encoding="utf-8") as f:
        for real, gen in zip(table.column("real").to_pylist(), table.column("generated").to_pylist()):
            if args.max_rows > 0 and n_ok >= args.max_rows:
                break
            rec_real = messages_to_record(real)
            rec_gen = messages_to_record(gen)
            if rec_real is None or rec_gen is None:
                n_skip += 1
                continue
            if rec_real["prompt"] != rec_gen["prompt"]:
                n_mismatch += 1
                continue
            f.write(json.dumps(rec_real, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Wrote {args.prompts_out} ({n_ok} lines)")
    print(f"skipped {n_skip} malformed, {n_mismatch} prompt-mismatched")


if __name__ == "__main__":
    main()
