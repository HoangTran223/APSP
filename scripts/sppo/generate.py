"""Sample K responses per prompt for one shard with vLLM (SPPO / S-SPPO / RSPO)."""
import os
os.environ.setdefault("WANDB_MODE", "offline")

import patch_transformers_torchao  # noqa: F401
import argparse
import random
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompt_utils import TEMPERATURE, TOP_P, apply_template, load_prompt_dataset, shard, write_json_atomic

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="data/Ultrachat50k/train.jsonl")
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--data_frac", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=1536)
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def keep_end(token_ids, limit, bos_id):
    """Left-truncate to `limit` tokens, keeping a leading BOS and the assistant header at the end."""
    if len(token_ids) <= limit:
        return token_ids
    if bos_id is not None and token_ids[:1] == [bos_id]:
        return [bos_id] + token_ids[len(token_ids) - (limit - 1):]
    return token_ids[len(token_ids) - limit:]


def main():
    args = parse_arguments()
    out_files = [os.path.join(args.output_dir, f"responses_{args.data_frac}_{p}.json") for p in range(args.pairs)]
    todo = [p for p, path in enumerate(out_files) if not os.path.exists(path)]
    if not todo:
        print(f"[generate] shard {args.data_frac} complete in {args.output_dir}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_prompt_dataset(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = shard([apply_template(text, tokenizer) for text in data["prompt"]], args.frac_len, args.data_frac)
    print(prompts[0])

    # Tokenize as vLLM would (special tokens included), then fit the context window.
    limit = args.max_model_len - args.maxlen
    bos_id = tokenizer.bos_token_id if getattr(tokenizer, "add_bos_token", False) else None
    token_prompts = []
    n_truncated = 0
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if len(ids) > limit:
            n_truncated += 1
            ids = keep_end(ids, limit, bos_id)
        token_prompts.append({"prompt_token_ids": ids})
    if n_truncated:
        print(f"[generate] left-truncated {n_truncated}/{len(prompts)} prompts to {limit} tokens")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
    )

    for p in todo:
        set_seed(p * 50)
        sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=args.maxlen, seed=p * 50)
        response = llm.generate(token_prompts, sampling_params)
        write_json_atomic(out_files[p], [r.outputs[0].text for r in response])
        print(f"[generate] wrote {out_files[p]}")


if __name__ == "__main__":
    main()
