"""Output-dir bookkeeping for the SPPO-family trainers: final weights, run fingerprint, completion mark."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

WEIGHT_GLOBS = ("model*.safetensors", "model*.bin", "pytorch_model*.bin")
RUN_CONFIG_NAME = "run_config.json"
COMPLETE_NAME = "training_complete.json"


def has_root_weights(output_dir: Union[str, Path]) -> bool:
    directory = Path(output_dir)
    if not directory.is_dir():
        return False
    for pattern in WEIGHT_GLOBS:
        if any(directory.glob(pattern)):
            return True
    return False


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _training_finished(state_path: Path) -> bool:
    with open(state_path) as f:
        state = json.load(f)
    global_step = int(state.get("global_step") or 0)
    max_steps = int(state.get("max_steps") or 0)
    epoch = float(state.get("epoch") or 0.0)
    num_epochs = float(state.get("num_train_epochs") or 0.0)
    if max_steps > 0 and global_step >= max_steps:
        return True
    if num_epochs > 0 and epoch >= num_epochs - 1e-6:
        return True
    return False


def drop_duplicate_final_checkpoint(
    output_dir: Union[str, Path],
    weights_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Delete the last checkpoint-* once the final weights exist in weights_dir (default output_dir)."""
    output_dir = Path(output_dir)
    if not has_root_weights(weights_dir or output_dir):
        return
    ckpts = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    ckpts.sort(key=_checkpoint_step)
    if not ckpts:
        return
    latest = ckpts[-1]
    state_path = latest / "trainer_state.json"
    if state_path.is_file() and not _training_finished(state_path):
        return
    shutil.rmtree(latest)
    logger.info(
        "Removed duplicate final checkpoint %s (weights already in %s)",
        latest,
        weights_dir or output_dir,
    )


class SkipLastEpochCheckpointMixin:
    """Skip checkpoint-* on the last epoch/step; save_model() writes the final copy."""

    def _is_last_epoch_checkpoint(self) -> bool:
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        global_step = int(getattr(self.state, "global_step", 0) or 0)
        if max_steps > 0 and global_step >= max_steps:
            return True
        num_epochs = float(getattr(self.args, "num_train_epochs", 0) or 0)
        epoch = float(getattr(self.state, "epoch", 0) or 0)
        return num_epochs > 0 and epoch >= num_epochs - 1e-6

    def _save_checkpoint(self, model, trial, metrics=None):
        if self._is_last_epoch_checkpoint():
            if self.is_world_process_zero():
                warnings.warn(
                    "Skipping checkpoint-* for the last epoch; final weights are written by save_model().",
                    UserWarning,
                    stacklevel=2,
                )
            return
        super_save = super()._save_checkpoint
        try:
            return super_save(model, trial, metrics=metrics)
        except TypeError:
            return super_save(model, trial)


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value):
    return getattr(value, "value", value)


def _resolved(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        return os.path.realpath(path)
    return path


def dataset_fingerprint(dataset_mixer: dict) -> dict:
    """Mixer fractions plus the sha256 of every train*.parquet in each local dataset dir."""
    out = {}
    for name, frac in sorted(dataset_mixer.items()):
        entry = {"frac": float(frac)}
        directory = Path(name)
        if directory.is_dir():
            entry["files"] = {f.name: file_sha256(f) for f in sorted(directory.glob("train*.parquet"))}
        out[_resolved(str(name))] = entry
    return out


def training_fingerprint(model_args, data_args, training_args, **extra) -> dict:
    fingerprint = {
        "model_name_or_path": _resolved(model_args.model_name_or_path),
        "datasets": dataset_fingerprint(data_args.dataset_mixer),
        "learning_rate": training_args.learning_rate,
        "lr_scheduler_type": _plain(training_args.lr_scheduler_type),
        "warmup_ratio": training_args.warmup_ratio,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "world_size": training_args.world_size,
        "optim": _plain(training_args.optim),
        "max_grad_norm": training_args.max_grad_norm,
        "seed": training_args.seed,
        "bf16": training_args.bf16,
        "beta": training_args.beta,
        "loss_type": training_args.loss_type,
        "max_length": training_args.max_length,
        "max_prompt_length": training_args.max_prompt_length,
    }
    fingerprint.update(extra)
    return json.loads(json.dumps(fingerprint, sort_keys=True))


def write_json_atomic(path: Union[str, Path], payload) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def prepare_output_dir(output_dir: Union[str, Path], fingerprint: dict, force: bool = False) -> str:
    """Return "done", "resume" or "fresh"; another configuration's output is refused unless force."""
    directory = Path(output_dir)
    if directory.is_dir() and any(directory.iterdir()):
        complete = _read_json(directory / COMPLETE_NAME)
        recorded = _read_json(directory / RUN_CONFIG_NAME)
        if complete is not None:
            if complete.get("fingerprint") == fingerprint and has_root_weights(directory):
                return "done"
        elif recorded == fingerprint:
            return "resume"
        if not force:
            raise SystemExit(
                f"{directory} holds a run with a different or unrecorded configuration. "
                "Delete it, or set FORCE_RETRAIN=1 to delete it and retrain."
            )
        logger.warning("Deleting %s (FORCE_RETRAIN=1).", directory)
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / RUN_CONFIG_NAME, fingerprint)
    return "fresh"


def mark_training_complete(output_dir: Union[str, Path], fingerprint: dict) -> None:
    write_json_atomic(Path(output_dir) / COMPLETE_NAME, {"fingerprint": fingerprint})
