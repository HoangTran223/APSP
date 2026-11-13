"""Prompt loading, chat formatting and sharding shared by the SPPO data scripts."""
import json
import os
from pathlib import Path

from datasets import load_dataset

RESPONSE_PLACEHOLDER = "None"
TEMPERATURE = 1.0
TOP_P = 1.0
RANKER = "llm-blender/PairRM"


def load_prompt_dataset(prompts_arg):
    if prompts_arg.endswith(".json") or prompts_arg.endswith(".jsonl"):
        return load_dataset("json", data_files=prompts_arg, split="train")
    return load_dataset(prompts_arg, split="train")


def apply_template(text, tokenizer):
    """Single-turn chat prompt up to the assistant reply, without the template's BOS (vLLM adds one)."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, {"role": "assistant", "content": RESPONSE_PLACEHOLDER}],
        tokenize=False,
    ).rsplit(RESPONSE_PLACEHOLDER, 1)[0]
    bos = tokenizer.bos_token
    if bos and getattr(tokenizer, "add_bos_token", False) and prompt.startswith(bos):
        prompt = prompt[len(bos):]
    return prompt


def shard(items, frac_len, data_frac):
    if frac_len <= 0:
        return list(items)
    return list(items[frac_len * data_frac: frac_len * (data_frac + 1)])


def num_shards(num_items, frac_len):
    return (num_items + frac_len - 1) // frac_len


def tmp_path(path):
    """Hidden sibling: datasets.load_dataset ignores dot-files, so a crash leaves no bogus split."""
    path = Path(path)
    return path.with_name(f".{path.name}.tmp")


def write_json_atomic(path, payload):
    tmp = tmp_path(path)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def write_parquet_atomic(df, path):
    tmp = tmp_path(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
