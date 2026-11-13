import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from checkpointing import COMPLETE_MARKER, LATEST_DIR, RESUME_META, TRAINER_STATE, model_identity, resume_matches, write_json_atomic
from preference_datasets import generation_bos_id, get_batch_iterator, get_dataset, tokenize_batch_element
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
)
import numpy as np
import wandb

import random
import os
import gc
import hashlib
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

import subprocess

REFERENCE_LOSSES = {'dpo', 'apsp', 'space'}


def _safe_save_pretrained(model, save_dir: str, **kwargs):
    try:
        model.save_pretrained(save_dir, **kwargs)
        return
    except ImportError as e:
        # accelerate's unwrap_model imports DeepSpeed even when it is unused.
        msg = str(e)
        if "torch.distributed.elastic.agent.server.api" not in msg:
            raise

    import accelerate.utils.other as accelerate_other
    old_checker = accelerate_other.is_deepspeed_available
    accelerate_other.is_deepspeed_available = lambda: False
    try:
        model.save_pretrained(save_dir, **kwargs)
    finally:
        accelerate_other.is_deepspeed_available = old_checker


def apsp_loss(policy_chosen_logps: torch.FloatTensor,
              policy_rejected_logps: torch.FloatTensor,
              reference_chosen_logps: torch.FloatTensor,
              reference_rejected_logps: torch.FloatTensor,
              precomputed_sign: torch.FloatTensor,
              beta: float) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """APSP: -log sigmoid(R(y+) + s R(y-))."""
    g_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    g_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
    sign = precomputed_sign.to(device=g_chosen.device, dtype=g_chosen.dtype).detach()
    margin = g_chosen + sign * g_rejected
    losses = -F.logsigmoid(margin)
    chosen_rewards = g_chosen.detach()
    rejected_rewards = (sign * g_rejected).detach()
    return losses, chosen_rewards, rejected_rewards, sign


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """DPO loss (SPIN's objective)."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def tspin_loss(policy_chosen_logps: torch.FloatTensor,
               policy_rejected_logps: torch.FloatTensor,
               policy_proto_logps: torch.FloatTensor,
               alpha: float = 1.0,
               beta: float = 0.1) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """T-SPIN: -logσ(α(log π(y) - log π(y'))) - β·logσ(α(log π(y') - log π(y_0)))."""
    u = alpha * (policy_chosen_logps - policy_rejected_logps)
    v = alpha * (policy_rejected_logps - policy_proto_logps)
    losses = -F.logsigmoid(u) - beta * F.logsigmoid(v)

    chosen_rewards = (alpha * policy_chosen_logps).detach()
    rejected_rewards = (alpha * policy_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def space_loss(policy_chosen_logps: torch.FloatTensor,
               policy_rejected_logps: torch.FloatTensor,
               reference_chosen_logps: torch.FloatTensor,
               reference_rejected_logps: torch.FloatTensor,
               mu: float = 1.0) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """SPACE NCE loss."""
    log_mu = torch.log(torch.as_tensor(mu, dtype=policy_chosen_logps.dtype, device=policy_chosen_logps.device))

    log_ratio_real = policy_chosen_logps - reference_chosen_logps
    real_loss = F.logsigmoid(log_ratio_real - log_mu)

    log_ratio_gen = reference_rejected_logps - policy_rejected_logps
    gen_loss = mu * F.logsigmoid(log_ratio_gen + log_mu)

    losses = -(real_loss + gen_loss)

    chosen_rewards = log_ratio_real.detach()
    rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model


def _backbone_and_head(model: nn.Module):
    inner = _unwrap_model(model)
    backbone = getattr(inner, 'model', None)
    head = getattr(inner, 'lm_head', None)
    if backbone is None or head is None:
        return None, None
    return backbone, head


def _apply_final_logit_softcap(logits: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Gemma-2 tanh softcap on the final logits."""
    cap = getattr(getattr(_unwrap_model(model), 'config', None), 'final_logit_softcapping', None)
    if cap is None:
        return logits
    cap = float(cap)
    logits = logits / cap
    logits = torch.tanh(logits)
    logits = logits * cap
    return logits


def _pair_key_from_id_bytes(chosen_ids_i32: bytes, rejected_ids_i32: bytes) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(chosen_ids_i32)
    h.update(b'\x00')
    h.update(rejected_ids_i32)
    return h.hexdigest()


def _ids_to_i32_bytes(ids) -> bytes:
    if torch.is_tensor(ids):
        arr = ids.detach()
        if arr.device.type != 'cpu':
            arr = arr.cpu()
        return arr.to(torch.int32).contiguous().numpy().tobytes()
    return np.asarray(ids, dtype=np.int32).tobytes()


def _pair_key_from_tensors(chosen_ids, chosen_mask, rejected_ids, rejected_mask) -> str:
    """Pair key over the unpadded token ids."""
    def _payload(ids, mask):
        ids_cpu = ids.detach()
        if ids_cpu.device.type != 'cpu':
            ids_cpu = ids_cpu.cpu()
        if mask is None:
            return ids_cpu.to(torch.int32).contiguous().numpy().tobytes()
        mask_cpu = mask.detach()
        if mask_cpu.device.type != 'cpu':
            mask_cpu = mask_cpu.cpu()
        return ids_cpu.to(torch.int32).numpy()[mask_cpu.bool().numpy()].tobytes()

    return _pair_key_from_id_bytes(_payload(chosen_ids, chosen_mask), _payload(rejected_ids, rejected_mask))


def _pad_stack(seqs, pad_value, dtype):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 0
    bsz = len(seqs)
    out = torch.full((bsz, max_len), pad_value, dtype=dtype)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, (s, n) in enumerate(zip(seqs, lengths)):
        if n == 0:
            continue
        if torch.is_tensor(s):
            out[i, :n] = s.to(dtype=dtype)
        else:
            out[i, :n] = torch.as_tensor(s, dtype=dtype)
        attn[i, :n] = 1
    return out, attn


def _token_packs(lengths, token_budget: int, max_bs: int):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    packs = []
    i = 0
    n = len(order)
    while i < n:
        j = i + 1
        max_len = max(1, int(lengths[order[i]]))
        while j < n:
            new_max = max(max_len, int(lengths[order[j]]))
            cnt = j - i + 1
            if cnt > max_bs or cnt * new_max > token_budget:
                break
            max_len = new_max
            j += 1
        packs.append(order[i:j])
        i = j
    return packs


def _get_batch_logps_from_hidden(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.LongTensor,
    model: nn.Module,
) -> torch.FloatTensor:
    """Summed response logps from the last hidden state, LM head applied in chunks."""
    assert hidden.shape[:2] == labels.shape
    labels = labels[:, 1:].clone()
    hidden = hidden[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    seq_len = labels.shape[1]
    bsz = labels.shape[0]
    vocab = int(getattr(lm_head, 'out_features', 0) or 256128)
    # Keep one logits chunk under ~1 GB.
    max_chunk = max(1, int(1.0e9 / (max(bsz, 1) * vocab * 2)))
    chunk = min(int(os.environ.get('LOGPS_CHUNK_SIZE', '64')), seq_len, max_chunk)
    pieces = []
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        logits = lm_head(hidden[:, start:end, :])
        logits = _apply_final_logit_softcap(logits, model)
        vocab = logits.size(-1)
        nll = F.cross_entropy(
            logits.reshape(-1, vocab),
            labels[:, start:end].reshape(-1),
            reduction='none',
        ).view(labels.shape[0], -1)
        pieces.append(-nll)
        del logits, nll
    per_token_logps = torch.cat(pieces, dim=-1)
    return (per_token_logps * loss_mask).sum(-1).float()


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """Sum of per-token logps. Labels of -100 are ignored."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab = logits.size(-1)
    chunk = min(int(os.environ.get('LOGPS_CHUNK_SIZE', '64')), labels.shape[1])
    pieces = []
    for start in range(0, labels.shape[1], chunk):
        end = min(start + chunk, labels.shape[1])
        nll = F.cross_entropy(
            logits[:, start:end, :].reshape(-1, vocab),
            labels[:, start:end].reshape(-1),
            reduction='none',
        ).view(labels.shape[0], -1)
        pieces.append(-nll)
    per_token_logps = torch.cat(pieces, dim=-1)
    return (per_token_logps * loss_mask).sum(-1).float()


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


def triplet_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    max_length = max(
        batch['chosen_input_ids'].shape[1],
        batch['rejected_input_ids'].shape[1],
        batch['proto_input_ids'].shape[1],
    )
    triplet_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            out_key = k.replace('chosen', 'concatenated')
            triplet_batch[out_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            out_key = k.replace('rejected', 'concatenated')
            triplet_batch[out_key] = torch.cat((
                triplet_batch[out_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    for k in batch:
        if k.startswith('proto') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            out_key = k.replace('proto', 'concatenated')
            triplet_batch[out_key] = torch.cat((
                triplet_batch[out_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return triplet_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, ckpt_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, run_fingerprint: Optional[dict] = None):
        self.seed = seed
        self.run_fingerprint = run_fingerprint
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.ckpt_dir = ckpt_dir
        self.base_data_dir = config.base_data_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        prepend_bos_id = generation_bos_id(self.tokenizer)
        self._prepend_bos_id = prepend_bos_id
        if prepend_bos_id is not None:
            rank0_print(f'Prepending generation BOS id {prepend_bos_id} inside max_length')
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            seed=seed,
            reverse_dataset=config.reverse_dataset,
            base_data_dir=config.base_data_dir,
            prepend_bos_id=prepend_bos_id,
        )

        self.policy = policy
        self.reference_model = reference_model
        self.data_iterator_kwargs = data_iterator_kwargs
        self.train_iterator = None

        # FSDPTrainer applies its own checkpoint wrapper.
        if config.activation_checkpointing and 'FSDP' not in str(config.trainer):
            for _m in (self.policy, self.reference_model):
                if _m is None:
                    continue
                if hasattr(_m, 'config'):
                    _m.config.use_cache = False
            if hasattr(self.policy, 'gradient_checkpointing_enable'):
                try:
                    self.policy.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={'use_reentrant': False}
                    )
                except TypeError:
                    self.policy.gradient_checkpointing_enable()
                if hasattr(self.policy, 'enable_input_require_grads'):
                    self.policy.enable_input_require_grads()
                rank0_print('Enabled HF gradient checkpointing on policy')

        self._ref_logps_cache: Optional[Dict[str, Tuple[float, float]]] = None

    def _latest_dir(self) -> str:
        return os.path.join(self.ckpt_dir, LATEST_DIR)

    def _save_latest_checkpoint(self):
        """resume_meta.json is removed first and written last, so a partial save is never resumed."""
        latest_dir = self._latest_dir()
        os.makedirs(latest_dir, exist_ok=True)
        meta_path = os.path.join(latest_dir, RESUME_META)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        print(f'Saving model to {latest_dir} ...')
        _safe_save_pretrained(self.policy, latest_dir, safe_serialization=True)
        try:
            self.tokenizer.save_pretrained(latest_dir)
        except Exception:
            pass
        state = {
            'example_counter': self.example_counter,
            'batch_counter': self.batch_counter,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        state_path = os.path.join(latest_dir, TRAINER_STATE)
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        write_json_atomic(meta_path, {
            'example_counter': self.example_counter,
            'batch_counter': self.batch_counter,
            'fingerprint': self.run_fingerprint,
        })
        rank0_print(f'Saved resume state at example_counter={self.example_counter}')

    def _load_resume_state(self) -> Optional[dict]:
        latest_dir = self._latest_dir()
        if not resume_matches(latest_dir, self.run_fingerprint):
            return None
        state_path = os.path.join(latest_dir, TRAINER_STATE)
        try:
            return torch.load(state_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(state_path, map_location='cpu')

    def _sequence_logps(self, model: nn.Module, input_ids, attention_mask, labels):
        device = next(model.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        backbone, lm_head = _backbone_and_head(model)
        if backbone is not None and lm_head is not None:
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = outputs.last_hidden_state
            logps = _get_batch_logps_from_hidden(hidden, lm_head, labels, model)
            del outputs, hidden
            return logps
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        logps = _get_batch_logps(logits, labels)
        del outputs, logits
        return logps

    def _ref_cache_path(self) -> str:
        return os.path.join(self.ckpt_dir, 'ref_logps_cache.pt')

    def _ref_cache_fingerprint(self) -> dict:
        files = []
        for name in self.config.datasets:
            fp = os.path.join(self.config.base_data_dir, name, 'train.jsonl')
            st = os.stat(fp)
            files.append({'path': os.path.abspath(fp), 'size': int(st.st_size), 'mtime_ns': int(st.st_mtime_ns)})
        fp = {
            'files': files,
            'max_length': int(self.config.max_length),
            'max_prompt_length': int(self.config.max_prompt_length),
            'tokenizer': str(self.config.model.tokenizer_name_or_path or self.config.model.name_or_path),
            'ref_model': model_identity(self.config.model.name_or_path),
        }
        if self._prepend_bos_id is not None:
            fp['prepend_bos_id'] = int(self._prepend_bos_id)
        return fp

    def _lookup_ref_logps(self, batch, device) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = batch['chosen_input_ids'].shape[0]
        chosen_vals, rejected_vals = [], []
        for i in range(bsz):
            key = _pair_key_from_tensors(
                batch['chosen_input_ids'][i], batch['chosen_attention_mask'][i],
                batch['rejected_input_ids'][i], batch['rejected_attention_mask'][i],
            )
            pair = self._ref_logps_cache.get(key)
            if pair is None:
                raise KeyError(f'ref logps cache miss for pair {key} (i={i}/{bsz})')
            chosen_vals.append(pair[0])
            rejected_vals.append(pair[1])
        return (
            torch.tensor(chosen_vals, device=device, dtype=torch.float32),
            torch.tensor(rejected_vals, device=device, dtype=torch.float32),
        )

    def _try_load_ref_logps_cache(self) -> bool:
        cache_path = self._ref_cache_path()
        if not os.path.isfile(cache_path):
            return False
        try:
            blob = torch.load(cache_path, map_location='cpu', weights_only=False)
        except TypeError:
            blob = torch.load(cache_path, map_location='cpu')
        if isinstance(blob, dict) and blob.get('fingerprint') == self._ref_cache_fingerprint() and blob.get('cache'):
            self._ref_logps_cache = blob['cache']
            rank0_print(f'Loaded frozen π_ref logps cache ({len(self._ref_logps_cache)} pairs) from {cache_path}')
            return True
        rank0_print(f'Ref logps cache fingerprint mismatch; will recompute ({cache_path})')
        return False

    def _precompute_ref_logps(self) -> None:
        """Frozen π_ref logps for every pair, computed once and reused across epochs."""
        if self._try_load_ref_logps_cache():
            return
        cache_path = self._ref_cache_path()

        if self.reference_model is None:
            raise RuntimeError('Cannot precompute π_ref logps: reference_model is None')

        token_budget = int(os.environ.get('REF_CACHE_TOKEN_BUDGET', '196608'))
        max_bs = int(os.environ.get('REF_CACHE_MAX_BS', '2048'))
        rank0_print(
            f'Precomputing frozen π_ref logps (token-pack budget={token_budget}, max_bs={max_bs}) -> {cache_path}'
        )
        self.reference_model.eval()

        elements = []
        for name in self.config.datasets:
            data = get_dataset(
                name, 'train', silent=self.rank != 0,
                base_data_dir=self.config.base_data_dir,
                reverse_dataset=bool(self.config.get('reverse_dataset', False)),
            )
            for prompt, rec in data.items():
                for pair in rec['pairs']:
                    elements.append(tokenize_batch_element(
                        prompt, rec['responses'][pair[0]], rec['responses'][pair[1]],
                        'keep_end', self.tokenizer,
                        int(self.config.max_length), int(self.config.max_prompt_length),
                        prepend_bos_id=self._prepend_bos_id,
                    ))
        n_el = len(elements)
        keys = [
            _pair_key_from_id_bytes(
                _ids_to_i32_bytes(el['chosen_input_ids']),
                _ids_to_i32_bytes(el['rejected_input_ids']),
            )
            for el in elements
        ]
        chosen_logps = [0.0] * n_el
        rejected_logps = [0.0] * n_el
        rank0_print(f'Tokenized {n_el} unique pairs; packed π_ref forwards')

        pad_id = int(self.tokenizer.pad_token_id)
        device = next(self.reference_model.parameters()).device

        def _score_side(prefix: str, out_list: list):
            lengths = [len(el[f'{prefix}_input_ids']) for el in elements]
            packs = _token_packs(lengths, token_budget, max_bs)
            done = 0
            t_side = time.time()
            with torch.inference_mode():
                for pi, idxs in enumerate(packs):
                    seqs = [elements[i][f'{prefix}_input_ids'] for i in idxs]
                    labs = [elements[i][f'{prefix}_labels'] for i in idxs]
                    input_ids, attn = _pad_stack(seqs, pad_id, torch.long)
                    labels, _ = _pad_stack(labs, -100, torch.long)
                    try:
                        logps = self._sequence_logps(
                            self.reference_model,
                            input_ids.to(device, non_blocking=True),
                            attn.to(device, non_blocking=True),
                            labels.to(device, non_blocking=True),
                        )
                    except torch.cuda.OutOfMemoryError:
                        del input_ids, attn, labels
                        torch.cuda.empty_cache()
                        if len(idxs) == 1:
                            raise
                        mid = max(1, len(idxs) // 2)
                        rank0_print(f'  OOM on {prefix} pack len={len(idxs)} T={seqs and max(len(s) for s in seqs)}; splitting to {mid}')
                        packs.append(idxs[:mid])
                        packs.append(idxs[mid:])
                        continue
                    vals = logps.detach().float().cpu().tolist()
                    for local_i, el_i in enumerate(idxs):
                        out_list[el_i] = float(vals[local_i])
                    done += len(idxs)
                    if (pi + 1) % 5 == 0 or done >= n_el:
                        elapsed = max(time.time() - t_side, 1e-6)
                        rank0_print(
                            f'  π_ref {prefix} {done}/{n_el} seqs, {done/elapsed:.1f} seq/s, '
                            f'pack_bs={len(idxs)} T={input_ids.shape[1]}'
                        )
                    del logps, input_ids, attn, labels

        t0 = time.time()
        _score_side('chosen', chosen_logps)
        _score_side('rejected', rejected_logps)
        cache: Dict[str, Tuple[float, float]] = {
            keys[i]: (chosen_logps[i], rejected_logps[i]) for i in range(n_el)
        }

        os.makedirs(self.ckpt_dir, exist_ok=True)
        tmp_path = cache_path + '.tmp'
        torch.save({'fingerprint': self._ref_cache_fingerprint(), 'cache': cache}, tmp_path)
        os.replace(tmp_path, cache_path)
        self._ref_logps_cache = cache
        elapsed = max(time.time() - t0, 1e-6)
        rank0_print(f'Cached {len(cache)} unique π_ref pairs in {elapsed:.1f}s ({len(cache)/elapsed:.1f} pairs/s) at {cache_path}')

    def _free_reference_model(self) -> None:
        if self.reference_model is None:
            return
        rank0_print('Freeing frozen π_ref weights (logps cache is live)')
        self.reference_model.to('cpu')
        del self.reference_model
        self.reference_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _batch_to_device(self, batch: Dict[str, Union[List, torch.LongTensor]], device: torch.device) -> Dict[str, Union[List, torch.LongTensor]]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

    def _reference_pair_logps(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if self._ref_logps_cache is not None:
            device = next(self.policy.parameters()).device
            return self._lookup_ref_logps(batch, device)
        ref_device = next(self.reference_model.parameters()).device
        policy_device = next(self.policy.parameters()).device
        ref_batch = batch if ref_device == policy_device else self._batch_to_device(batch, ref_device)
        chosen_logps, rejected_logps = self.concatenated_forward(self.reference_model, ref_batch)
        if chosen_logps.device != policy_device:
            chosen_logps = chosen_logps.to(policy_device)
            rejected_logps = rejected_logps.to(policy_device)
        return chosen_logps, rejected_logps

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.world_size == 1 and os.environ.get('SEQUENTIAL_FORWARD', '0') == '1':
            chosen_logps = self._sequence_logps(
                model, batch['chosen_input_ids'], batch['chosen_attention_mask'],
                batch['chosen_labels'],
            )
            rejected_logps = self._sequence_logps(
                model, batch['rejected_input_ids'], batch['rejected_attention_mask'],
                batch['rejected_labels'],
            )
            return chosen_logps, rejected_logps

        concatenated_batch = concatenated_inputs(batch)
        all_logps = self._sequence_logps(
            model,
            concatenated_batch['concatenated_input_ids'],
            concatenated_batch['concatenated_attention_mask'],
            concatenated_batch['concatenated_labels'],
        )
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def triplet_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self.world_size == 1 and os.environ.get('SEQUENTIAL_FORWARD', '0') == '1':
            chosen_logps = self._sequence_logps(
                model, batch['chosen_input_ids'], batch['chosen_attention_mask'],
                batch['chosen_labels'],
            )
            rejected_logps = self._sequence_logps(
                model, batch['rejected_input_ids'], batch['rejected_attention_mask'],
                batch['rejected_labels'],
            )
            proto_logps = self._sequence_logps(
                model, batch['proto_input_ids'], batch['proto_attention_mask'],
                batch['proto_labels'],
            )
            return chosen_logps, rejected_logps, proto_logps

        triplet_batch = triplet_inputs(batch)
        all_logps = self._sequence_logps(
            model,
            triplet_batch['concatenated_input_ids'],
            triplet_batch['concatenated_attention_mask'],
            triplet_batch['concatenated_labels'],
        )
        n_chosen = batch['chosen_input_ids'].shape[0]
        n_rejected = batch['rejected_input_ids'].shape[0]
        chosen_logps = all_logps[:n_chosen]
        rejected_logps = all_logps[n_chosen:n_chosen + n_rejected]
        proto_logps = all_logps[n_chosen + n_rejected:]
        return chosen_logps, rejected_logps, proto_logps

    def _side_backward(self, batch: Dict[str, Union[List, torch.LongTensor]], side: str, coeff: torch.Tensor, accum: float):
        """Backward of coeff * logps for one side, reduced like losses.mean() / accum."""
        if float(coeff.abs().max().item()) == 0.0:
            return
        logps = self._sequence_logps(
            self.policy,
            batch[f'{side}_input_ids'], batch[f'{side}_attention_mask'],
            batch[f'{side}_labels'],
        )
        ((logps * coeff).mean() / accum).backward()
        del logps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'apsp':
            ss_lowmem = (
                bool(train)
                and self.world_size == 1
                and os.environ.get('DPO_LOWMEM', '0') == '1'
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self._reference_pair_logps(batch)
                if ss_lowmem:
                    policy_rejected_logps = self._sequence_logps(
                        self.policy, batch['rejected_input_ids'], batch['rejected_attention_mask'],
                        batch['rejected_labels'],
                    )
            if ss_lowmem and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ss_lowmem:
                policy_chosen_logps = self._sequence_logps(
                    self.policy, batch['chosen_input_ids'], batch['chosen_attention_mask'],
                    batch['chosen_labels'],
                )
            else:
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)

            raw_signs = batch.get('apsp_sign', None)
            if raw_signs is None or any(s is None for s in raw_signs):
                raise ValueError('loss=apsp needs apsp_sign on every pair; score the round with score_apsp.py')
            precomputed_sign = torch.tensor(
                [float(s) for s in raw_signs],
                device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype,
            )

            losses, chosen_rewards, rejected_rewards, sign = apsp_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                precomputed_sign,
                beta=loss_config.beta,
            )

            if ss_lowmem:
                if not getattr(self, '_logged_apsp_lowmem', False):
                    rank0_print('DPO_LOWMEM=1: per-side APSP policy backward')
                    self._logged_apsp_lowmem = True
                beta = float(loss_config.beta)
                # d/d(policy rejected logp) of -log sigmoid(R+ + s R-), R- = beta * (logp - ref).
                logits = (
                    beta * (policy_chosen_logps.detach() - reference_chosen_logps)
                    + sign * beta * (policy_rejected_logps - reference_rejected_logps)
                )
                dL_dr = (-torch.sigmoid(-logits)) * (sign * beta)
                accum = float(self.config.gradient_accumulation_steps)
                (losses.mean() / accum).backward()
                policy_chosen_logps = policy_chosen_logps.detach()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._side_backward(batch, 'rejected', dL_dr.detach(), accum)
                losses = losses.detach()

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
            sign = all_gather_if_needed(sign, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'apsp_{train_test}/frac_positive'] = [(sign == 1).float().mean().item()]
            metrics[f'apsp_{train_test}/frac_negative'] = [(sign == -1).float().mean().item()]
            metrics[f'apsp_{train_test}/count_positive'] = [(sign == 1).sum().item()]
            metrics[f'apsp_{train_test}/count_negative'] = [(sign == -1).sum().item()]
            metrics[f'apsp_{train_test}/count_total'] = [float(sign.numel())]

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'tspin':
            policy_chosen_logps, policy_rejected_logps, policy_proto_logps = self.triplet_forward(self.policy, batch)

            losses, chosen_rewards, rejected_rewards = tspin_loss(
                policy_chosen_logps, policy_rejected_logps, policy_proto_logps,
                alpha=loss_config.alpha,
                beta=loss_config.beta,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            proto_rewards = (loss_config.alpha * policy_proto_logps).detach()
            proto_rewards = all_gather_if_needed(proto_rewards, self.rank, self.world_size)
            metrics[f'rewards_{train_test}/proto'] = proto_rewards.cpu().numpy().tolist()

            policy_rejected_logps_gathered = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps_gathered.cpu().numpy().tolist()
            policy_proto_logps_gathered = all_gather_if_needed(policy_proto_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/proto'] = policy_proto_logps_gathered.cpu().numpy().tolist()

        elif loss_config.name == 'space':
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            losses, chosen_rewards, rejected_rewards = space_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                mu=loss_config.mu,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'dpo':
            # Ref first (no grad) so its activations/logits are freed before the policy graph.
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            if reference_chosen_logps.device != policy_chosen_logps.device:
                reference_chosen_logps = reference_chosen_logps.to(policy_chosen_logps.device, non_blocking=True)
                reference_rejected_logps = reference_rejected_logps.to(policy_chosen_logps.device, non_blocking=True)

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
                beta=loss_config.beta, label_smoothing=loss_config.label_smoothing,
                reference_free=loss_config.reference_free,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.float().cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.float().cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.float().cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).float().cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.float().cpu().numpy().tolist()

        else:
            raise ValueError(f'unknown loss {loss_config.name}')

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.float().cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.float().cpu().numpy().tolist()

        return losses.mean(), metrics

    def _count_training_examples(self) -> int:
        total = 0
        for name in self.config.datasets:
            fp = os.path.join(self.config.base_data_dir, name, 'train.jsonl')
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    total += sum(1 for _ in f)
            except FileNotFoundError:
                rank0_print(f'WARNING: dataset file not found while sizing LR schedule: {fp}')
        return total

    def train(self):
        cache_ref = (
            self.config.loss.name == 'apsp'
            and os.environ.get('CACHE_REF_LOGPS', '1') != '0'
            and self.world_size == 1
        )
        if cache_ref:
            if not self._try_load_ref_logps_cache():
                if self.reference_model is None:
                    raise RuntimeError('Cannot precompute π_ref logps: no cache and reference_model is None')
                # Park policy so the frozen-ref pass can use a larger batch without OOM.
                policy_device = next(self.policy.parameters()).device
                self.policy.to('cpu')
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._precompute_ref_logps()
                self.policy.to(policy_device)
            self._free_reference_model()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)

        warmup_ratio = self.config.get('warmup_ratio', None)
        if warmup_ratio is not None and self.config.n_epochs is not None:
            n_examples = self._count_training_examples()
            steps_per_epoch = max(1, n_examples // self.config.batch_size)
            total_steps = steps_per_epoch * self.config.n_epochs
            warmup_steps = max(1, round(warmup_ratio * total_steps))
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            rank0_print(f'LR schedule: linear warmup+decay | N={n_examples}, batch={self.config.batch_size}, '
                        f'epochs={self.config.n_epochs} -> total_steps={total_steps}, warmup_steps={warmup_steps} (ratio={warmup_ratio})')
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
            rank0_print(f'LR schedule: warmup->constant | warmup_steps={self.config.warmup_steps}')

        resume_state = self._load_resume_state() if self.config.get('resume', True) else None
        if resume_state is not None:
            self.optimizer.load_state_dict(resume_state['optimizer'])
            self.scheduler.load_state_dict(resume_state['scheduler'])
            self.example_counter = int(resume_state['example_counter'])
            self.batch_counter = int(resume_state['batch_counter'])
            rank0_print(f'Restored optimizer+scheduler; continuing at example_counter={self.example_counter}')
        else:
            self.example_counter = 0
            self.batch_counter = 0
        skip = self.example_counter

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if (not cache_ref) and self.reference_model is not None and self.config.loss.name in REFERENCE_LOSSES:
            self.reference_model.eval()

        self.train_iterator = get_batch_iterator(
            **self.data_iterator_kwargs,
            split='train',
            n_epochs=self.config.n_epochs,
            n_examples=self.config.n_examples,
            batch_size=self.config.batch_size,
            silent=self.rank != 0,
            skip_examples=skip,
        )
        rank0_print(f'Loaded train data iterator (skip_examples={skip})')

        last_log = None
        just_resumed = skip > 0

        apsp_sign_counts = {'positive': 0, 'negative': 0, 'total': 0}

        for batch in self.train_iterator:
            if (not just_resumed) and self.example_counter > 0 and self.example_counter % 2000 == 0 and self.rank == 0:
                self._save_latest_checkpoint()

                eval_script = os.environ.get('EVAL_SCRIPT', '').strip()
                if not eval_script:
                    print('EVAL_SCRIPT is not set, skipping external evaluation.')
                elif not os.path.isfile(eval_script):
                    print(f'EVAL_SCRIPT not found ({eval_script}), skipping external evaluation.')
                else:
                    print(f'Evaluating with script: {eval_script}')
                    try:
                        eval_env = os.environ.copy()
                        eval_env['EVAL_MODEL_PATH'] = self._latest_dir()
                        result = subprocess.run(
                            ['bash', eval_script],
                            check=True,
                            capture_output=True,
                            text=True,
                            env=eval_env,
                        )
                        if result.stdout:
                            print(result.stdout)
                            if wandb.run is not None:
                                wandb.log({'eval/stdout': result.stdout})
                        print('External evaluation finished successfully.')
                    except subprocess.CalledProcessError as e:
                        print('External evaluation failed; training will continue.')
                        if e.stdout:
                            print(e.stdout)
                        if e.stderr:
                            print(e.stderr)

            just_resumed = False

            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                if loss.requires_grad:
                    (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            if self.config.loss.name == 'apsp':
                apsp_sign_counts['positive'] += int(sum(batch_metrics.get('apsp_train/count_positive', [0])))
                apsp_sign_counts['negative'] += int(sum(batch_metrics.get('apsp_train/count_negative', [0])))
                apsp_sign_counts['total'] += int(sum(batch_metrics.get('apsp_train/count_total', [0])))

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if self.example_counter % 50 == 0:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

            if self.config.loss.name == 'apsp' and self.example_counter % 10000 == 0:
                tot = apsp_sign_counts['total'] or 1
                rank0_print(
                    f'[APSP sign stats] after {self.example_counter} examples: '
                    f'+1={apsp_sign_counts["positive"]} ({100*apsp_sign_counts["positive"]/tot:.1f}%), '
                    f'-1={apsp_sign_counts["negative"]} ({100*apsp_sign_counts["negative"]/tot:.1f}%), '
                    f'total={apsp_sign_counts["total"]}'
                )

        if self.config.loss.name == 'apsp':
            tot = apsp_sign_counts['total'] or 1
            summary = {
                'count_positive': apsp_sign_counts['positive'],
                'count_negative': apsp_sign_counts['negative'],
                'count_total': apsp_sign_counts['total'],
                'frac_positive': apsp_sign_counts['positive'] / tot,
                'frac_negative': apsp_sign_counts['negative'] / tot,
                'config': {
                    'beta': float(self.config.loss.beta),
                },
            }
            rank0_print('=' * 60)
            rank0_print('[APSP] Final sign distribution for this iteration:')
            rank0_print(f'  +1 (promote) : {summary["count_positive"]:>7d}  ({100*summary["frac_positive"]:.2f}%)')
            rank0_print(f'  -1 (demote)  : {summary["count_negative"]:>7d}  ({100*summary["frac_negative"]:.2f}%)')
            rank0_print(f'  total        : {summary["count_total"]:>7d}')
            rank0_print('=' * 60)

            sign_stats_path = os.path.join(self.ckpt_dir, 'apsp_sign_stats.json')
            os.makedirs(self.ckpt_dir, exist_ok=True)
            with open(sign_stats_path, 'w') as f:
                json.dump(summary, f, indent=2)
            rank0_print(f'[APSP] Sign statistics saved to {sign_stats_path}')


    def clip_gradient(self):
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def _begin_final_save(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        marker = os.path.join(self.ckpt_dir, COMPLETE_MARKER)
        if os.path.exists(marker):
            os.remove(marker)

    def _mark_complete(self):
        write_json_atomic(os.path.join(self.ckpt_dir, COMPLETE_MARKER), {
            'example_counter': self.example_counter,
            'batch_counter': self.batch_counter,
            'fingerprint': self.run_fingerprint,
        })

    def save(self):
        self._begin_final_save()
        _safe_save_pretrained(self.policy, self.ckpt_dir)
        rank0_print(f"Model saved to {self.ckpt_dir} using save_pretrained")
        self.tokenizer.save_pretrained(self.ckpt_dir)
        self._mark_complete()


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, ckpt_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, run_fingerprint: Optional[dict] = None):
        super().__init__(policy, config, seed, run_dir, ckpt_dir, reference_model, rank, world_size, run_fingerprint=run_fingerprint)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in REFERENCE_LOSSES:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self._begin_final_save()
            from transformers import AutoModelForCausalLM
            unwrapped_model = AutoModelForCausalLM.from_pretrained(self.config.model.name_or_path)
            unwrapped_model.load_state_dict(policy_state_dict)
            _safe_save_pretrained(unwrapped_model, self.ckpt_dir)
            rank0_print(f"Model saved to {self.ckpt_dir} using save_pretrained")
            del unwrapped_model
            self.tokenizer.save_pretrained(self.ckpt_dir)
            self._mark_complete()

        del policy_state_dict
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, ckpt_dir: str, reference_model=None, rank=0, world_size=1, run_fingerprint=None):
        import tensor_parallel as tp

        super().__init__(policy, config, seed, run_dir, ckpt_dir, reference_model, rank, world_size, run_fingerprint=run_fingerprint)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in REFERENCE_LOSSES:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self):
        import tensor_parallel as tp

        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self._begin_final_save()
        from transformers import AutoModelForCausalLM
        unwrapped_model = AutoModelForCausalLM.from_pretrained(self.config.model.name_or_path)
        unwrapped_model.load_state_dict(policy_state_dict)
        _safe_save_pretrained(unwrapped_model, self.ckpt_dir)
        rank0_print(f"Model saved to {self.ckpt_dir} using save_pretrained")
        del unwrapped_model
        self.tokenizer.save_pretrained(self.ckpt_dir)
        self._mark_complete()
        del policy_state_dict
