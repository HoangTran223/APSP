import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, build_exp_name
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import checkpointing
import trainers
import wandb
import socket
from typing import Optional, Set
import resource

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
OmegaConf.register_new_resolver("build_exp_name", lambda loss_name, model_name, datasets, reverse_dataset: build_exp_name(loss_name, model_name, datasets, reverse_dataset))


def _load_causal_lm(name_or_path: str, dtype: torch.dtype, model_kwargs: dict):
    base_kwargs = dict(low_cpu_mem_usage=True, **model_kwargs)
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            name_or_path,
            dtype=dtype,
            **base_kwargs,
        )
    except TypeError:
        return transformers.AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=dtype,
            **base_kwargs,
        )


def worker_main(rank: int, world_size: int, config: DictConfig, run_fingerprint: dict, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, config.ckpt_dir, reference_model=reference_model,
                           rank=rank, world_size=world_size, run_fingerprint=run_fingerprint)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="configs/main", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    ckpt_dir = str(config.ckpt_dir)
    attn_impl = os.environ.get('ATTN_IMPLEMENTATION', 'sdpa')
    run_fingerprint = checkpointing.run_fingerprint(config, attn_impl)
    state = checkpointing.checkpoint_state(ckpt_dir, run_fingerprint, config.get('resume', True))
    if state == 'stale' and os.environ.get('FORCE_RETRAIN', '0') == '1':
        print(f'[train] FORCE_RETRAIN=1: deleting the previous run in {ckpt_dir}')
        checkpointing.clear_run_files(ckpt_dir)
        state = 'fresh'
    if state == 'done':
        print(f'[train] skip finished iteration: {ckpt_dir}')
        return
    if state == 'stale':
        raise SystemExit(
            f'[train] {ckpt_dir} holds weights, LATEST/ or a completion marker from a different or unrecorded run. '
            'Delete them, or set FORCE_RETRAIN=1 to delete them and retrain.'
        )
    latest_dir = os.path.join(ckpt_dir, checkpointing.LATEST_DIR)
    load_from_latest = state == 'resume'
    if load_from_latest:
        print(f'[train] resuming from {latest_dir}')

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.output_dir)
    print('building policy')
    # device_map reserves most of each card, so it is used only to split across GPUs.
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_kwargs = {}
    if config.trainer == 'BasicTrainer' and n_visible > 1:
        model_kwargs['device_map'] = 'balanced'
    if attn_impl:
        model_kwargs['attn_implementation'] = attn_impl
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy_src = latest_dir if load_from_latest else config.model.name_or_path
    print(f'policy weights: {policy_src}')
    policy = _load_causal_lm(policy_src, policy_dtype, model_kwargs)
    if 'device_map' not in model_kwargs and torch.cuda.is_available():
        policy = policy.to('cuda')
        print(f'[train] policy on cuda (no device_map; {n_visible} visible GPU)')
    if hasattr(policy, 'config'):
        policy.config.use_cache = False
    disable_dropout(policy)

    if config.loss.name in trainers.REFERENCE_LOSSES:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = _load_causal_lm(config.model.name_or_path, reference_model_dtype, model_kwargs)
        if os.environ.get('SPIN_REF_CPU', '0') == '1':
            reference_model = reference_model.to('cpu')
            print('[train] reference model on CPU (SPIN_REF_CPU=1)')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif 'device_map' not in model_kwargs and torch.cuda.is_available():
            reference_model = reference_model.to('cuda')
        if hasattr(reference_model, 'config'):
            reference_model.config.use_cache = False
        disable_dropout(reference_model)
    else:
        reference_model = None

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, run_fingerprint, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, run_fingerprint, policy, reference_model)


if __name__ == '__main__':
    main()