"""Sample one response per prompt with vLLM and write {prompt, chosen, rejected} rows."""
import os

# torch.compile during vLLM startup is unstable on this stack.
os.environ.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
import time

from checkpointing import file_sha256, model_identity, read_json, write_json_atomic

TEMPERATURE = 1.0
TOP_P = 1.0
# SPIN prompt template.
PROMPT_PREFIX = "### Instruction: "
PROMPT_SUFFIX = "\n\n### Response: "


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True, help='prompt jsonl with prompt and ground_truth fields')
    parser.add_argument('--output_dir', type=str, required=True, help='writes <output_dir>.jsonl')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--world_size', type=int, default=1, help='vLLM tensor-parallel size')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=4096, help='vLLM context window (prompt + response)')
    parser.add_argument('--max_num_seqs', type=int, default=128,
                        help='Concurrent sequences (continuous batching). Lower if OOM.')
    parser.add_argument('--max_num_batched_tokens', type=int, default=16384,
                        help='Prefill/decode token budget per scheduler step.')
    parser.add_argument('--no_enforce_eager', action='store_true',
                        help='Allow CUDA graphs / compiled paths (usually faster). Omit if vLLM crashes.')
    return parser.parse_args()


def keep_end(token_ids, limit, bos_id):
    """Left-truncate to `limit` tokens, keeping a leading BOS."""
    if len(token_ids) <= limit:
        return token_ids
    if bos_id is not None and token_ids[:1] == [bos_id]:
        return [bos_id] + token_ids[len(token_ids) - (limit - 1):]
    return token_ids[len(token_ids) - limit:]


def generate(args, prompts):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=not args.no_enforce_eager,
        disable_log_stats=True,
    )
    # Tokenize as vLLM would (special tokens included), then fit the context window.
    limit = max(1, args.max_model_len - args.max_new_tokens)
    inputs = [{"prompt_token_ids": keep_end(tokenizer(p).input_ids, limit, tokenizer.bos_token_id)} for p in prompts]
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=args.max_new_tokens)

    start = time.time()
    outputs = llm.generate(inputs, sampling_params)
    print(f"time elapsed: {time.time() - start:.1f}s")
    return [o.outputs[0].text.replace("</s>", "").lstrip() for o in outputs]


def main():
    args = parse_arguments()
    out_path = f"{args.output_dir}.jsonl"
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    spec = json.loads(json.dumps({
        "model": model_identity(args.model),
        "input": {"path": os.path.realpath(args.input_dir), "sha256": file_sha256(args.input_dir)},
        "rows": [0, len(data)],
        "template": [PROMPT_PREFIX, PROMPT_SUFFIX],
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }, sort_keys=True))
    if os.path.isfile(out_path) and read_json(meta_path) == spec:
        print(f"[generate_vllm] up to date: {out_path}")
        return
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    prompts = [PROMPT_PREFIX + d['prompt'] + PROMPT_SUFFIX for d in data]
    responses = generate(args, prompts)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for d, prompt, response in zip(data, prompts, responses):
            f.write(json.dumps({"prompt": prompt, "chosen": d['ground_truth'], "rejected": response}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)
    write_json_atomic(meta_path, spec)
    print(f"[generate_vllm] wrote {len(data)} rows -> {out_path}")


if __name__ == "__main__":
    main()
