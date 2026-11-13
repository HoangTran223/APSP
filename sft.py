#!/usr/bin/env python3
# SFT on HuggingFaceH4/ultrachat_200k; the loss covers every assistant turn.

import os
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "SPPO" / "sppo"))
from save_utils import SkipLastEpochCheckpointMixin, drop_duplicate_final_checkpoint  # noqa: E402


class _SFTTrainer(SkipLastEpochCheckpointMixin, Trainer):
    pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sft")

# template marker -> (assistant turn opener, turn end, merge system into user)
CHAT_FORMATS = {
    "<|im_start|>": ("<|im_start|>assistant\n", "<|im_end|>", False),
    "<start_of_turn>": ("<start_of_turn>model\n", "<end_of_turn>", True),
}


def chat_format(tokenizer):
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError("the tokenizer defines no chat template")
    for marker, fmt in CHAT_FORMATS.items():
        if marker in template:
            return fmt
    raise ValueError("unsupported chat template (expected ChatML or Gemma turn markers)")


def merge_system(messages: List[dict]) -> List[dict]:
    """Merge any system message into the following user turn."""
    out: List[dict] = []
    pending_system = None
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            pending_system = content.strip()
            continue
        if role == "user" and pending_system:
            content = (pending_system + "\n\n" + content).strip()
            pending_system = None
        out.append({"role": role, "content": content})
    return out


def assistant_char_spans(text: str, head: str, end: str) -> List[tuple]:
    """Char spans of every assistant response, its end-of-turn token included."""
    spans, i = [], 0
    while True:
        h = text.find(head, i)
        if h == -1:
            break
        cstart = h + len(head)
        e = text.find(end, cstart)
        cend = len(text) if e == -1 else e + len(end)
        spans.append((cstart, cend))
        i = cend
    return spans


def build_tokenize_fn(tokenizer, max_seq_len: int):
    head, end, system_to_user = chat_format(tokenizer)

    def _tok(batch: Dict[str, list]) -> Dict[str, list]:
        out_ids, out_labels = [], []
        for messages in batch["messages"]:
            if system_to_user:
                messages = merge_system(messages)
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                continue  # the template rejects the conversation (e.g. roles do not alternate)

            enc = tokenizer(
                text,
                add_special_tokens=False,  # the template already emits BOS where the model has one
                truncation=True,
                max_length=max_seq_len,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"]
            labels = [-100] * len(input_ids)

            spans = assistant_char_spans(text, head, end)
            si = 0
            for idx, (a, b) in enumerate(enc["offset_mapping"]):
                if b <= a:
                    continue
                while si < len(spans) and a >= spans[si][1]:
                    si += 1
                if si < len(spans):
                    s, e = spans[si]
                    if a >= s and b <= e:
                        labels[idx] = input_ids[idx]

            if all(lab == -100 for lab in labels):
                continue  # truncation removed every assistant token
            out_ids.append(input_ids)
            out_labels.append(labels)
        return {"input_ids": out_ids, "labels": out_labels}

    return _tok


def load_splits(args):
    if args.dataset_path and os.path.isdir(args.dataset_path):
        logger.info(f"Loading dataset from disk: {args.dataset_path}")
        ds = load_from_disk(args.dataset_path)
        train_ds = ds[args.train_split]
        eval_ds = ds[args.eval_split] if args.eval_split in ds else None
    else:
        logger.info(f"Loading dataset from hub: {args.dataset_name}")
        train_ds = load_dataset(args.dataset_name, "default", split=args.train_split)
        try:
            eval_ds = load_dataset(args.dataset_name, "default", split=args.eval_split)
        except Exception:
            eval_ds = None

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if eval_ds is not None and args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = load_splits(args)
    tok_fn = build_tokenize_fn(tokenizer, args.max_seq_len)

    logger.info("Tokenizing + masking train split ...")
    train_tok = train_ds.map(
        tok_fn, batched=True, batch_size=1000,
        num_proc=args.num_proc, remove_columns=train_ds.column_names,
        desc="tokenize-train",
    )
    eval_tok = None
    if eval_ds is not None:
        logger.info("Tokenizing + masking eval split ...")
        eval_tok = eval_ds.map(
            tok_fn, batched=True, batch_size=1000,
            num_proc=args.num_proc, remove_columns=eval_ds.column_names,
            desc="tokenize-eval",
        )

    logger.info(f"Train examples after masking/filtering: {len(train_tok)}")
    if eval_tok is not None:
        logger.info(f"Eval examples: {len(eval_tok)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding="longest", label_pad_token_id=-100,
        pad_to_multiple_of=8, return_tensors="pt",
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        adam_beta1=0.9,
        adam_beta2=0.999,
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        eval_strategy=("epoch" if eval_tok is not None else "no"),
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = _SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("=== Starting training ===")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving final model ...")
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    drop_duplicate_final_checkpoint(args.output_dir, weights_dir=final_dir)
    logger.info(f"Done. Final model at {final_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="SFT on ultrachat_200k")
    # paths
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, default="data/ultrachat_200k",
                   help="load_from_disk dir; if missing, falls back to --dataset_name")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--train_split", type=str, default="train_sft")
    p.add_argument("--eval_split", type=str, default="test_sft")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # data/tokenization
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--num_proc", type=int, default=8)

    # optimization
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")

    # system
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="Gemma-2 needs eager: SDPA and FlashAttention-2 skip its logit soft-capping.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # logging / checkpoints
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--report_to", type=str, default="wandb", help="'none' disables logging integrations")
    p.add_argument("--run_name", type=str, default="sft-ultrachat200k")

    return p.parse_args()


if __name__ == "__main__":
    main()
