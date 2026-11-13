"""Completion and resume records for train.py checkpoints, keyed by a run fingerprint."""
import hashlib
import json
import os
import shutil

from omegaconf import OmegaConf

COMPLETE_MARKER = 'training_complete.json'
RESUME_META = 'resume_meta.json'
TRAINER_STATE = 'trainer_state.pt'
LATEST_DIR = 'LATEST'
WEIGHT_FILES = ('model.safetensors', 'model.safetensors.index.json')
WEIGHT_SUFFIXES = ('.safetensors', '.safetensors.index.json', '.bin', '.bin.index.json')

RUN_KEYS = (
    'seed', 'datasets', 'reverse_dataset', 'lr', 'batch_size', 'gradient_accumulation_steps',
    'max_grad_norm', 'max_length', 'max_prompt_length', 'n_epochs', 'n_examples', 'trainer',
    'optimizer', 'warmup_steps', 'warmup_ratio', 'loss', 'model',
)


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json_atomic(path, payload):
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(name_or_path):
    """Resolved path plus weight-file sizes and mtimes; a hub id is kept as is."""
    if not os.path.isdir(name_or_path):
        return str(name_or_path)
    root = os.path.realpath(name_or_path)
    weights = {}
    for name in sorted(os.listdir(root)):
        if name.endswith(WEIGHT_SUFFIXES):
            stat = os.stat(os.path.join(root, name))
            weights[name] = [stat.st_size, stat.st_mtime_ns]
    return {'path': root, 'weights': weights}


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def run_fingerprint(config, attn_implementation):
    fingerprint = {key: _plain(config.get(key)) for key in RUN_KEYS}
    fingerprint['attn_implementation'] = attn_implementation
    fingerprint['policy_init'] = model_identity(config.model.name_or_path)
    fingerprint['data'] = {
        name: file_sha256(os.path.join(config.base_data_dir, name, 'train.jsonl')) for name in config.datasets
    }
    return json.loads(json.dumps(fingerprint, sort_keys=True))


def has_weights(model_dir):
    return any(os.path.isfile(os.path.join(model_dir, name)) for name in WEIGHT_FILES)


def _run_files(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return []
    return [name for name in os.listdir(ckpt_dir) if name == COMPLETE_MARKER or name.endswith(WEIGHT_SUFFIXES)]


def resume_matches(latest_dir, fingerprint):
    meta = read_json(os.path.join(latest_dir, RESUME_META))
    return (
        meta is not None and meta.get('fingerprint') == fingerprint
        and has_weights(latest_dir) and os.path.isfile(os.path.join(latest_dir, TRAINER_STATE))
    )


def checkpoint_state(ckpt_dir, fingerprint, resume):
    """'done', 'resume' (from LATEST), 'fresh' (nothing written yet) or 'stale' (another run's files)."""
    marker = read_json(os.path.join(ckpt_dir, COMPLETE_MARKER))
    if marker is not None and marker.get('fingerprint') == fingerprint and has_weights(ckpt_dir):
        return 'done'
    latest_dir = os.path.join(ckpt_dir, LATEST_DIR)
    if resume and resume_matches(latest_dir, fingerprint):
        return 'resume'
    if _run_files(ckpt_dir) or os.path.isdir(latest_dir):
        return 'stale'
    return 'fresh'


def clear_run_files(ckpt_dir):
    """Delete the weights, LATEST/ and completion marker of a previous run; keep everything else."""
    shutil.rmtree(os.path.join(ckpt_dir, LATEST_DIR), ignore_errors=True)
    for name in _run_files(ckpt_dir):
        os.remove(os.path.join(ckpt_dir, name))
