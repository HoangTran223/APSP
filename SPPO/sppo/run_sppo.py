#!/usr/bin/env python
# Adapted from https://github.com/huggingface/alignment-handbook

import inspect
import logging
import os
import sys
from pathlib import Path

# Must run before transformers / peft are imported.
_scripts_sppo = Path(__file__).resolve().parents[2] / "scripts" / "sppo"
if str(_scripts_sppo) not in sys.path:
    sys.path.insert(0, str(_scripts_sppo))
import patch_transformers_torchao  # noqa: F401

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    SPPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from peft import PeftConfig, PeftModel
from trainer import SPPOTrainer
from save_utils import (
    drop_duplicate_final_checkpoint,
    has_root_weights,
    mark_training_complete,
    prepare_output_dir,
    training_fingerprint,
)

logger = logging.getLogger(__name__)


def patch_accelerate_unwrap_model():
    # transformers passes keep_torch_compile, which older accelerate does not accept.
    try:
        from accelerate import Accelerator
    except Exception:
        return

    sig = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in sig.parameters:
        return

    original_unwrap_model = Accelerator.unwrap_model

    def compat_unwrap_model(self, model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return original_unwrap_model(self, model, *args, **kwargs)

    Accelerator.unwrap_model = compat_unwrap_model


patch_accelerate_unwrap_model()


def setup_logging(log_level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def load_and_process_datasets(data_args, tokenizer):
    raw_datasets = get_datasets(data_args, splits=["train"])
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names = [x for x in column_names if x not in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose', 'sim_score']]

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "skip_system_message": True},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    return raw_datasets


def setup_model(model_args, training_args):
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    # Under bf16 training load bf16 weights; fp32 weights, grads and RMSprop state do not fit.
    if torch_dtype is None and getattr(training_args, "bf16", False):
        torch_dtype = torch.bfloat16
        logger.info("Loading model in bfloat16 (training_args.bf16=True).")
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    elif model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        if model_args.attn_implementation is not None:
            model_kwargs["attn_implementation"] = model_args.attn_implementation
        elif model_args.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    return model, model_kwargs


def _usable_checkpoint(path) -> bool:
    if not path:
        return False
    directory = Path(path)
    return (directory / "trainer_state.json").is_file() and has_root_weights(directory)


def train_and_evaluate(trainer, raw_datasets, training_args, checkpoint=None):
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def save_model_and_results(trainer, training_args, model_args, data_args):
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    if trainer.accelerator.is_main_process:
        drop_duplicate_final_checkpoint(training_args.output_dir)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    trainer.accelerator.wait_for_everyone()
    logger.info("*** Training complete! ***")


def resolve_checkpoint(training_args):
    if training_args.resume_from_checkpoint is not None:
        return training_args.resume_from_checkpoint
    checkpoint = get_checkpoint(training_args)
    if checkpoint is not None and not _usable_checkpoint(checkpoint):
        logger.warning("Ignoring incomplete checkpoint %s (missing weights or trainer_state.json).", checkpoint)
        return None
    return checkpoint


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPPOConfig))
    model_args, data_args, training_args = parser.parse()
    training_args.do_eval = False
    setup_logging(training_args.get_process_log_level())

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    fingerprint = training_fingerprint(model_args, data_args, training_args, lam=training_args.lam)
    state = prepare_output_dir(
        training_args.output_dir, fingerprint, force=os.environ.get("FORCE_RETRAIN", "0") == "1"
    )
    if state == "done":
        logger.info(f"{training_args.output_dir} already finished with this configuration; skipping training.")
        return
    checkpoint = resolve_checkpoint(training_args) if state == "resume" else None
    if checkpoint is not None:
        logger.info(f"Resuming training at {checkpoint=}.")

    set_seed(training_args.seed)

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)
    raw_datasets = load_and_process_datasets(data_args, tokenizer)

    model, model_kwargs = setup_model(model_args, training_args)

    # The frozen reference is the starting policy; its logps are precomputed, so no second copy is kept.
    trainer = SPPOTrainer(
        model,
        None,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=None,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        lam=training_args.lam,
        precompute_ref_log_probs=True,
    )

    train_and_evaluate(trainer, raw_datasets, training_args, checkpoint=checkpoint)
    save_model_and_results(trainer, training_args, model_args, data_args)
    if trainer.accelerator.is_main_process:
        mark_training_complete(training_args.output_dir, fingerprint)


if __name__ == "__main__":
    main()
