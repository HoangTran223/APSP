<h1 align="center">APSP</h1>
<p align="center"><strong>Adaptive Polarity Self-Play</strong></p>
<p align="center"><em>Not All Rollouts Are Negative: Adaptive Polarity in Self-Play Fine-Tuning</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.4%20%2B%20cu121-ee4c2c?style=flat-square" alt="PyTorch 2.4 cu121">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2f80ed?style=flat-square" alt="License"></a>
</p>

Official code for **APSP** (Adaptive Polarity Self-Play), from *Not All Rollouts Are Negative: Adaptive Polarity in Self-Play Fine-Tuning*.
This repository contains the APSP training code and the self-play baselines it is compared against.

## Methods included

| Method | Description                                |
|--------|--------------------------------------------|
| APSP   | Adaptive Polarity Self-Play                |
| SPIN   | Self-Play Fine-Tuning                      |
| SPPO   | Self-Play Preference Optimization          |
| SPACE  | Self-Play via Noise Contrastive Estimation |
| RSPO   | Regularized SPO, reverse-KL variant        |
| S-SPPO | Semantic-Calibrated SPPO                   |
| T-SPIN | Triplet SPIN                               |

## Install

```bash
# Qwen2.5-1.5B stack (Py 3.10, torch 2.4 / cu121, transformers 4.45, vLLM 0.6.3)
conda env create -f environment.yml
conda activate apsp
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# Qwen3-4B stack (transformers 4.51, vLLM 0.8.5), kept separate from the env above;
# scripts/sft.sh uses it for all three models
conda env create -f environment-qwen3.yml
conda activate apsp-qwen3
```

## Data preparation

```bash
# UltraChat50k self-play prompts -> data/Ultrachat50k/train.jsonl
huggingface-cli download UCLA-AGI/SPIN_iter0 --repo-type dataset \
  --local-dir data/SPIN_iter0
python scripts/prepare_spin_iter0.py \
  --parquet data/SPIN_iter0/train_prefs-00000-of-00001.parquet \
  --prompts_out data/Ultrachat50k/train.jsonl

# Optional, for nodes without hub access: UltraChat200k for the SFT stage
python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/ultrachat_200k').save_to_disk('data/ultrachat_200k')"
```

## Base models and SFT

Download each pretrained model to its path below. `bash scripts/sft.sh <model>` fine-tunes it and
writes `output/*_sft_ultrachat200k_<date>/final`; move that directory to the SFT model path.

| `<model>` | Pretrained model     | Download to               | SFT model                                  |
|-----------|----------------------|---------------------------|--------------------------------------------|
| `qwen2.5` | `Qwen/Qwen2.5-1.5B`  | `model_hub/Qwen2.5-1.5B`  | `model_hub/Qwen2.5-1.5B-ultrachat200k`     |
| `gemma2`  | `google/gemma-2-2b`  | `model_hub/gemma-2-2b`    | `model_hub/gemma-2-2b-ultrachat200k/final` |
| `qwen3`   | `Qwen/Qwen3-4B-Base` | `model_hub/Qwen3-4B-Base` | `model_hub/Qwen3-4B-ultrachat200k`         |

## Running a method

```bash
# bash scripts/<METHOD>.sh [qwen2.5|gemma2|qwen3]    qwen2.5 by default
GPU_IDS=0 bash scripts/APSP.sh
GPU_IDS=0 bash scripts/SPIN.sh
GPU_IDS=0 bash scripts/SPPO.sh
GPU_IDS=0 bash scripts/SPACE.sh
GPU_IDS=0 bash scripts/RSPO.sh
GPU_IDS=0 bash scripts/S_SPPO.sh
GPU_IDS=0 bash scripts/T_SPIN.sh

GPU_IDS=0 bash scripts/APSP.sh gemma2
```

APSP / SPIN / SPACE / T-SPIN write their checkpoints to `model_hub/*-ultrachat200k/<METHOD>/iteX/`,
SPPO / S-SPPO to `SPPO/checkpoints/` and RSPO to `RSPO/checkpoints/`. Other settings (`TAU`, `BETA`,
`MAX_ITE`, `ITER_NUM`, `LR_PEAK`, ...) are environment variables read by `scripts/spin/run_iterations.sh`
and `scripts/sppo/run_iterations.sh`.

## Repository layout

```
APSP/
├── scripts/<METHOD>.sh   one entry point per method (APSP, SPIN, SPPO, SPACE, RSPO, S_SPPO, T_SPIN)
├── scripts/models.sh     per-model paths and settings, sourced by the method scripts
├── scripts/sft.sh        SFT entry point
├── scripts/prepare_spin_iter0.py   UltraChat50k prompt preparation
├── scripts/spin/         APSP / SPIN / SPACE / T-SPIN iteration driver
├── scripts/sppo/         SPPO / S-SPPO / RSPO iteration driver and data pipeline
├── train.py              trainer behind APSP / SPIN / SPACE / T-SPIN
├── trainers.py           the loss implementations it dispatches to
├── generate_vllm.py      self-play response generation for those four
├── score_apsp.py         APSP scoring: discrepancy d against p_base and the sign s per pair
├── checkpointing.py      completion / resume records that gate train.py checkpoints
├── sft.py                SFT stage on UltraChat200k (all three models)
├── configs/main/         hydra config for train.py (model / loss groups)
├── configs/sppo,rspo/    recipes consumed by the vendored SPPO / RSPO trainers
├── SPPO/                 vendored SPPO trainer (sppo/run_sppo.py + local alignment/)
├── RSPO/                 vendored RSPO trainer (sppo/run_sppo_reg.py + local alignment/)
├── model_hub/            base + SFT + self-play checkpoints (created by the steps above)
├── data/                 UltraChat50k prompts, optional UltraChat200k copy (created above)
├── environment.yml, requirements.txt              Qwen2.5-1.5B stack
├── environment-qwen3.yml, requirements-qwen3.txt  Qwen3-4B stack
└── README.md
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
