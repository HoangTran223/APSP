#!/usr/bin/env python3
"""APSP discrepancy scoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from checkpointing import file_sha256, model_identity, read_json, write_json_atomic
from preference_datasets import generation_bos_id, prepend_bos_token, truncate_preference_tokens

_SNR_VAR_SAFE = 1e-4
_SNR_GAP_REAL = 1e-2
_SNR_ROW_CAP = 512

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def snr_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Per-position SNR; a = p_base, b = policy, [N, V] -> [N]."""
    log_a = F.log_softmax(logits_a.float(), dim=-1)
    log_b = F.log_softmax(logits_b.float(), dim=-1)
    a = log_a.exp()
    h_a = -(a * log_a).sum(dim=-1)
    second = (a * log_a.square()).sum(dim=-1)
    del a
    b = log_b.exp()
    h_b = -(b * log_b).sum(dim=-1)
    kl_b_a = (b * (log_b - log_a)).sum(dim=-1)
    del b, log_a, log_b
    mean_gap = -kl_b_a + h_a - h_b
    var = second - h_a.square()
    snr = mean_gap.square() / var.clamp_min(torch.finfo(var.dtype).tiny)
    keep = (var > _SNR_VAR_SAFE) | (mean_gap.abs() > _SNR_GAP_REAL)
    return torch.where(keep, snr, torch.zeros_like(snr))


def discrepancy_from_snr(snr: torch.Tensor) -> float:
    if snr.numel() == 0:
        return float("inf")
    return float(snr.mean().item())


def apsp_sign_from_d(discrepancy: float, tau: float) -> int:
    if not np.isfinite(discrepancy):
        return -1
    return 1 if discrepancy <= tau else -1


def response_logit_span(length: int, response_start: int) -> tuple[int, int]:
    """Logit positions [start, end) that predict the response tokens, eos included."""
    if length <= 0 or response_start >= length:
        return 0, 0
    start = max(int(response_start) - 1, 0)
    end = int(length) - 1
    if end <= start:
        return 0, 0
    return start, end


_logits_to_keep_ok: dict[int, bool] = {}


def _snr_row_step(chunk: int, batch: int) -> int:
    return max(1, min(_SNR_ROW_CAP, max(1, int(chunk)) * max(1, int(batch))))


def snr_rows_from_batch_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    spans: list[tuple[int, int]],
    chunk: int,
) -> list[torch.Tensor]:
    if logits_b.device != logits_a.device:
        logits_b = logits_b.to(logits_a.device, non_blocking=logits_b.is_cuda)
    lengths = [max(0, int(e) - int(s)) for s, e in spans]
    total = sum(lengths)
    bsz = int(logits_a.shape[0])
    vocab = int(logits_a.shape[-1])
    if total == 0:
        empty = torch.empty(0, dtype=torch.float32, device=logits_a.device)
        return [empty.clone() for _ in range(bsz)]
    flat_a = logits_a.new_empty((total, vocab))
    flat_b = logits_b.new_empty((total, vocab))
    offset = 0
    for i, ((start, end), n) in enumerate(zip(spans, lengths)):
        if n <= 0:
            continue
        flat_a[offset:offset + n].copy_(logits_a[i, start:end])
        flat_b[offset:offset + n].copy_(logits_b[i, start:end])
        offset += n
    del logits_a, logits_b
    step = _snr_row_step(chunk, bsz)
    pieces = []
    for start in range(0, total, step):
        end = min(start + step, total)
        pieces.append(snr_from_logits(flat_a[start:end], flat_b[start:end]))
    flat = torch.cat(pieces, dim=0)
    del flat_a, flat_b, pieces
    rows: list[torch.Tensor] = []
    offset = 0
    for n in lengths:
        rows.append(flat[offset:offset + n])
        offset += n
    return rows


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _score_attn(path: str) -> str:
    forced = os.environ.get("APSP_SCORE_ATTN", "").strip()
    if forced:
        return forced
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "") or "")
    softcap = getattr(cfg, "attn_logit_softcapping", None) or getattr(cfg, "final_logit_softcapping", None)
    if "gemma" in model_type or softcap:
        return "eager"
    env = os.environ.get("ATTN_IMPLEMENTATION", "").strip()
    if env:
        return env
    return "sdpa"


def _load(path: str, dtype: torch.dtype, device: torch.device, attn: str):
    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except Exception as exc:
        if attn in {"sdpa", "eager"}:
            raise
        logger.warning("attn_implementation=%s failed (%s); retrying sdpa", attn, exc)
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        attn = "sdpa"
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    logger.info("loaded %s on %s attn=%s", path, device, attn)
    return model


def _pick_devices(policy_s: str, base_s: str) -> tuple[torch.device, torch.device]:
    if not torch.cuda.is_available():
        cpu = torch.device("cpu")
        return cpu, cpu
    n = torch.cuda.device_count()
    policy = torch.device((policy_s or "cuda:0").strip() or "cuda:0")
    if base_s.strip():
        base = torch.device(base_s.strip())
    elif n >= 2 and policy.type == "cuda":
        pol_index = 0 if policy.index is None else int(policy.index)
        base_index = 0 if pol_index != 0 else 1
        base = policy if base_index >= n or base_index == pol_index else torch.device("cuda", base_index)
    else:
        base = policy
    return policy, base


def _forward_logits(model, device: torch.device, ids: torch.Tensor, mask: torch.Tensor, keep: torch.Tensor | None):
    ids =ids.to(device, non_blocking=ids.is_pinned() if ids.device.type == "cpu" else False)
    mask = mask.to(device, non_blocking=mask.is_pinned() if mask.device.type == "cpu" else False)
    keep_dev = None if keep is None else keep.to(device, non_blocking=True)
    with torch.inference_mode():
        key = id(model)
        if keep_dev is not None and _logits_to_keep_ok.get(key, True):
            try:
                logits = model(
                    input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep_dev,
                ).logits
                _logits_to_keep_ok[key] = True
                return logits
            except TypeError:
                _logits_to_keep_ok[key] = False
                logger.info("model has no logits_to_keep; scoring the full sequence")
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits


class _DeviceWorker:
    """Runs one model's forwards on its own device thread, so p_base and the policy overlap."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self._q: list = []
        self._cv = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name=f"apsp-score-{device}", daemon=True)
        self._thread.start()

    def _loop(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait()
                if not self._q:
                    return
                item = self._q.pop(0)
            if item is None:
                return
            ids, mask, keep, slot = item
            try:
                slot["logits"] = _forward_logits(self.model, self.device, ids, mask, keep)
            except BaseException as exc:
                slot["error"] = exc
                if _is_oom(exc) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                slot["event"].set()

    def forward(self, ids, mask, keep):
        slot = {"event": threading.Event(), "logits": None, "error": None}
        with self._cv:
            self._q.append((ids, mask, keep, slot))
            self._cv.notify()
        return slot

    def close(self):
        with self._cv:
            self._closed = True
            self._q.append(None)
            self._cv.notify()
        self._thread.join()


def _pad_batch(rows: list[tuple[int, list[int], int]], pad_id: int):
    lengths = [len(ids) for _idx, ids, _rs in rows]
    width = max(lengths)
    bsz = len(rows)
    ids = torch.full((bsz, width), pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, width), dtype=torch.long)
    spans = []
    for i, ((_idx, tok, response_start), length) in enumerate(zip(rows, lengths)):
        ids[i, :length] = torch.tensor(tok, dtype=torch.long)
        mask[i, :length] = 1
        spans.append(response_logit_span(length, response_start))
    if torch.cuda.is_available():
        ids = ids.pin_memory()
        mask = mask.pin_memory()
    return ids, mask, spans, lengths


def _narrow_logits(logits: torch.Tensor, seq_len: int, t_lo: int, t_hi: int) -> torch.Tensor:
    width = t_hi - t_lo
    if int(logits.shape[1]) == width:
        return logits
    if int(logits.shape[1]) == seq_len:
        window = logits[:, t_lo:t_hi, :].contiguous()
        del logits
        return window
    raise RuntimeError(
        f"logits length {int(logits.shape[1])} is neither the sequence {seq_len} nor the response window {width}"
    )


def _score_padded_batch(policy, base, policy_worker, base_worker, ids, mask, spans, chunk: int):
    active = [span for span in spans if span[1] > span[0]]
    if not active:
        return [ids.new_empty((0,), dtype=torch.float32) for _ in spans]
    t_lo = min(s for s, e in active)
    t_hi = max(e for s, e in active)
    keep = torch.arange(t_lo, t_hi, dtype=torch.long)
    if policy_worker is None:
        logits_b = _forward_logits(policy, next(policy.parameters()).device, ids, mask, keep)
        logits_a = _forward_logits(base, next(base.parameters()).device, ids, mask, keep)
    else:
        slot_b = policy_worker.forward(ids, mask, keep)
        slot_a = base_worker.forward(ids, mask, keep)
        slot_b["event"].wait()
        slot_a["event"].wait()
        if slot_b["error"] is not None:
            raise slot_b["error"]
        if slot_a["error"] is not None:
            raise slot_a["error"]
        logits_b = slot_b["logits"]
        logits_a = slot_a["logits"]
        slot_b["logits"] = None
        slot_a["logits"] = None
    seq_len = int(ids.shape[1])
    logits_a = _narrow_logits(logits_a, seq_len, t_lo, t_hi)
    logits_b = _narrow_logits(logits_b, seq_len, t_lo, t_hi)
    local = [(s - t_lo, e - t_lo) if e > s else (0, 0) for s, e in spans]
    rows = snr_rows_from_batch_logits(logits_a, logits_b, local, chunk)
    del logits_a, logits_b
    return rows


def _length_batches(rows: list, batch_size: int) -> list[list]:
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
    return [[rows[i] for i in order[s:s + batch_size]] for s in range(0, len(order), batch_size)]


def _encode_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int,
    chosen: str | None = None,
    prepend_bos_id: int | None = None,
    truncation_mode: str = "keep_end",
):
    """Prompt + synthetic response tokens, truncated as preference_datasets.tokenize_batch_element does."""
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    rejected_tokens = tokenizer(response, add_special_tokens=False)
    chosen_tokens = tokenizer(chosen if chosen is not None else response, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        rejected_tokens["input_ids"] = rejected_tokens["input_ids"] + [eos]
        rejected_tokens["attention_mask"] = rejected_tokens["attention_mask"] + [1]
        chosen_tokens["input_ids"] = chosen_tokens["input_ids"] + [eos]
        chosen_tokens["attention_mask"] = chosen_tokens["attention_mask"] + [1]
    prompt_tokens, chosen_tokens, rejected_tokens, _proto = truncate_preference_tokens(
        prompt_tokens, chosen_tokens, rejected_tokens, None,
        truncation_mode, max_length, max_prompt_length,
    )
    if prepend_bos_id is not None:
        response_len = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
        prompt_tokens = prepend_bos_token(prompt_tokens, int(prepend_bos_id), response_len, max_length)
    prompt_ids = prompt_tokens["input_ids"]
    resp_ids = rejected_tokens["input_ids"]
    return prompt_ids + resp_ids, len(prompt_ids)


def _release_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_rows(policy, base, policy_worker, base_worker, rows, pad_id: int, chunk: int):
    if not rows:
        return []
    try:
        ids, mask, spans, _lengths = _pad_batch(rows, pad_id)
        snr_rows = _score_padded_batch(policy, base, policy_worker, base_worker, ids, mask, spans, chunk)
        out = []
        for (idx, _tok, _rs), snr in zip(rows, snr_rows):
            out.append((idx, discrepancy_from_snr(snr)))
        del ids, mask, snr_rows
        return out
    except RuntimeError as exc:
        if not _is_oom(exc) or len(rows) == 1:
            raise
        logger.info("OOM on score batch %d; splitting", len(rows))
        _release_cuda()
        mid = max(1, len(rows) // 2)
        return (
            _score_rows(policy, base, policy_worker, base_worker, rows[:mid], pad_id, chunk)
            + _score_rows(policy, base, policy_worker, base_worker, rows[mid:], pad_id, chunk)
        )


def main():
    parser = argparse.ArgumentParser(description="APSP scoring")
    parser.add_argument("--base_model", type=str, required=True, help="p_base: the frozen SFT model")
    parser.add_argument("--policy_model", type=str, required=True, help="pi_t: the policy that sampled the responses")
    parser.add_argument("--input_jsonl", type=str, required=True, help="generate_vllm.py output")
    parser.add_argument("--output_jsonl", type=str, required=True, help="input rows plus apsp_d and apsp_sign")
    parser.add_argument("--tau", type=float, required=True, help="s = +1 when d <= tau")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=int(os.environ.get("APSP_SNR_CHUNK", "64")))
    parser.add_argument("--score_batch", type=int, default=int(os.environ.get("APSP_SCORE_BATCH", "8")))
    parser.add_argument("--policy_device", type=str, default=os.environ.get("APSP_POLICY_DEVICE", ""))
    parser.add_argument("--base_device", type=str, default=os.environ.get("APSP_BASE_DEVICE", ""))
    parser.add_argument("--batch_log_interval", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    output_path = args.output_jsonl
    if os.path.realpath(output_path) == os.path.realpath(args.input_jsonl):
        parser.error("--output_jsonl must differ from --input_jsonl")
    meta_path = os.path.splitext(output_path)[0] + ".meta.json"
    spec = json.loads(json.dumps({
        "input": {"path": os.path.realpath(args.input_jsonl), "sha256": file_sha256(args.input_jsonl)},
        "base_model": model_identity(args.base_model),
        "policy_model": model_identity(args.policy_model),
        "tau": args.tau,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "dtype": args.dtype,
    }, sort_keys=True))
    if os.path.isfile(output_path) and read_json(meta_path) == spec:
        logger.info("up to date: %s", output_path)
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    same_model = os.path.realpath(args.base_model) == os.path.realpath(args.policy_model)
    logger.info("p_base=%s", args.base_model)
    logger.info("pi_t=%s same_weights=%s", args.policy_model, same_model)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos_id = generation_bos_id(tokenizer)
    logger.info("generation_bos_id=%s", bos_id)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]
    logger.info(
        "examples=%d score_batch=%d chunk=%d tau=%s",
        len(examples), args.score_batch, args.chunk, args.tau,
    )

    scores: list[float] = [float("inf")] * len(examples)
    signs: list[int] = [-1] * len(examples)
    pending = []
    for idx, ex in enumerate(examples):
        full_ids, response_start = _encode_pair(
            tokenizer, ex["prompt"], ex["rejected"], args.max_length, args.max_prompt_length,
            chosen=ex.get("chosen"), prepend_bos_id=bos_id,
        )
        if response_start >= len(full_ids):
            continue
        if same_model:
            scores[idx] = 0.0
            signs[idx] = apsp_sign_from_d(0.0, args.tau)
            continue
        pending.append((idx, full_ids, response_start))

    policy = base = None
    policy_worker = base_worker = None
    if pending:
        policy_dev, base_dev = _pick_devices(args.policy_device, args.base_device)
        attn_policy = _score_attn(args.policy_model)
        attn_base = _score_attn(args.base_model)
        policy = _load(args.policy_model, torch_dtype, policy_dev, attn_policy)
        base = _load(args.base_model, torch_dtype, base_dev, attn_base)
        parallel = policy_dev != base_dev
        logger.info(
            "score devices policy=%s p_base=%s parallel=%s",
            policy_dev, base_dev, parallel,
        )
        if parallel:
            policy_worker = _DeviceWorker(policy, policy_dev)
            base_worker = _DeviceWorker(base, base_dev)
        pad_id = int(tokenizer.pad_token_id or 0)
        scored = 0
        try:
            batches = _length_batches(pending, max(1, args.score_batch))
            pbar = tqdm(total=len(pending), desc="APSP d(x)")
            for batch in batches:
                got = _score_rows(
                    policy, base, policy_worker, base_worker, batch, pad_id, args.chunk,
                )
                for idx, discrepancy in got:
                    scores[idx] = discrepancy
                    signs[idx] = apsp_sign_from_d(discrepancy, args.tau)
                scored += len(batch)
                pbar.update(len(batch))
                if args.batch_log_interval > 0 and scored % args.batch_log_interval < len(batch):
                    finite = [s for s in scores if np.isfinite(s)]
                    n_pos = sum(1 for s in signs if s == 1)
                    mean = float(np.mean(finite)) if finite else float("nan")
                    logger.info("[%d/%d] d mean=%.6f | s=+1 %d/%d", scored, len(pending), mean, n_pos, len(examples))
            pbar.close()
        finally:
            if policy_worker is not None:
                policy_worker.close()
            if base_worker is not None:
                base_worker.close()

    n_pos = sum(1 for s in signs if s == 1)
    logger.info("s=+1 %d  s=-1 %d", n_pos, len(signs) - n_pos)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for ex, score, sign in zip(examples, scores, signs):
            ex["apsp_d"] = round(score, 8) if np.isfinite(score) else None
            ex["apsp_sign"] = sign
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    write_json_atomic(meta_path, spec)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
