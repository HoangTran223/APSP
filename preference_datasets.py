import datasets
import torch
import warnings

warnings.filterwarnings('ignore', message='Failed to load image Python extension')

from utils import TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union
import json


def get_dataset(name: str, split: str, silent: bool = False, base_data_dir: str = None, reverse_dataset: bool = False):
    """Load a jsonl preference file into {prompt: {responses, pairs, apsp_sign, proto}}."""
    file_path = f'{base_data_dir}/{name}/{split}.jsonl'
    print(f'Loading {name} dataset ({split} split) from {file_path}...')
    data = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc=f'Processing {name}', disable=silent):
            example = json.loads(line)
            prompt = example['prompt']
            chosen = example['chosen']
            rejected = example['rejected']
            apsp_sign = example.get('apsp_sign', None)
            proto = example.get('proto', None)  # T-SPIN's y_0

            if reverse_dataset:
                chosen, rejected = rejected, chosen

            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend([chosen, rejected])

            if apsp_sign is not None:
                if not isinstance(data[prompt].get('apsp_sign'), list):
                    data[prompt]['apsp_sign'] = []
                data[prompt]['apsp_sign'].append(apsp_sign)
            elif 'apsp_sign' not in data[prompt]:
                data[prompt]['apsp_sign'] = None

            if proto is not None:
                if not isinstance(data[prompt].get('proto'), list):
                    data[prompt]['proto'] = []
                data[prompt]['proto'].append(proto)
            elif 'proto' not in data[prompt]:
                data[prompt]['proto'] = None

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    def collate_fn(batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                else:
                    padding_value = 0

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def generation_bos_id(tokenizer) -> Optional[int]:
    """BOS id that vLLM prepends when it encodes a completion prompt, or None."""
    if not getattr(tokenizer, "add_bos_token", False):
        return None
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return None
    return int(bos)


def truncate_preference_tokens(prompt_tokens, chosen_tokens, rejected_tokens, proto_tokens, truncation_mode, max_length, max_prompt_length):
    if proto_tokens is None:
        longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
    else:
        longer_response_length = max(
            len(chosen_tokens['input_ids']),
            len(rejected_tokens['input_ids']),
            len(proto_tokens['input_ids']),
        )

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        cap = max_length - max_prompt_length
        chosen_tokens = {k: v[:cap] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:cap] for k, v in rejected_tokens.items()}
        if proto_tokens is not None:
            proto_tokens = {k: v[:cap] for k, v in proto_tokens.items()}
    return prompt_tokens, chosen_tokens, rejected_tokens, proto_tokens


def prepend_bos_token(prompt_tokens, bos_id: int, response_len: int, max_length: int):
    """Prepend BOS, dropping the oldest prompt tokens only if it would exceed max_length."""
    ids = prompt_tokens['input_ids']
    if ids and ids[0] == bos_id:
        return prompt_tokens
    drop = min(len(ids), max(0, len(ids) + 1 + response_len - max_length))
    if drop:
        prompt_tokens = {k: v[drop:] for k, v in prompt_tokens.items()}
    bos_value = {'input_ids': int(bos_id), 'attention_mask': 1}
    return {k: [bos_value.get(k, 0)] + list(v) for k, v in prompt_tokens.items()}


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int, proto: Optional[str] = None, prepend_bos_id: Optional[int] = None) -> Dict:
    """Tokenize one pair, plus T-SPIN's proto response as a third side if given."""
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    proto_tokens = None
    if proto is not None:
        proto_tokens = tokenizer(proto, add_special_tokens=False)

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    if proto_tokens is not None:
        proto_tokens['input_ids'].append(tokenizer.eos_token_id)
        proto_tokens['attention_mask'].append(1)

    prompt_tokens, chosen_tokens, rejected_tokens, proto_tokens = truncate_preference_tokens(
        prompt_tokens, chosen_tokens, rejected_tokens, proto_tokens,
        truncation_mode, max_length, max_prompt_length,
    )
    if prepend_bos_id is not None:
        response_len = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
        if proto_tokens is not None:
            response_len = max(response_len, len(proto_tokens['input_ids']))
        prompt_tokens = prepend_bos_token(prompt_tokens, int(prepend_bos_id), response_len, max_length)

    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    if proto_tokens is not None:
        proto_sequence_tokens = {k: prompt_tokens[k] + proto_tokens[k] for k in proto_tokens}
        proto_sequence_tokens['labels'] = proto_sequence_tokens['input_ids'][:]
        proto_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    else:
        proto_sequence_tokens = None

    batch = {}
    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected
    if proto is not None:
        batch['proto'] = prompt + proto
        batch['proto_response_only'] = proto

    toks_map = {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}
    if proto_sequence_tokens is not None:
        toks_map['proto'] = proto_sequence_tokens
    for k, toks in toks_map.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       base_data_dir: Optional[str] = None,
                       reverse_dataset: bool = False,
                       skip_examples: int = 0,
                       prepend_bos_id: Optional[int] = None) -> Iterator[Dict]:
    """Iterate batches until n_epochs or n_examples; skip_examples skips already-trained batches."""
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end'
            for prompt, data in get_dataset(name, split, silent=silent, base_data_dir=base_data_dir, reverse_dataset=reverse_dataset).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data.get('apsp_sign', None), data.get('proto', None), truncation_mode))

    collate_fn = get_collate_fn(tokenizer)
    epoch_idx = 0
    yielded_examples = 0
    done = False
    if skip_examples and not silent:
        print(f'Skipping first {skip_examples} examples (resume)')
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        pending = []
        for prompt, responses, pairs, apsp_sign, proto, truncation_mode in flat_data:
            if done:
                break
            for index, p in enumerate(pairs):
                if done:
                    break
                pending.append((prompt, responses[p[0]], responses[p[1]], truncation_mode,
                                apsp_sign[index] if apsp_sign else None,
                                proto[index] if proto else None))
                if len(pending) < batch_size:
                    continue
                if yielded_examples < skip_examples:
                    yielded_examples += batch_size
                    pending = []
                    continue
                batch = []
                for (pr, ch, rj, trunc, sg, proto_item) in pending:
                    batch_element = tokenize_batch_element(pr, ch, rj, trunc, tokenizer, max_length, max_prompt_length, proto=proto_item, prepend_bos_id=prepend_bos_id)
                    if sg is not None:
                        batch_element['apsp_sign'] = sg
                    batch.append(batch_element)
                yield collate_fn(batch)
                yielded_examples += batch_size
                pending = []
                if n_examples is not None and yielded_examples >= n_examples:
                    if not silent:
                        print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                    done = True
        if done:
            break

        epoch_idx += 1

