# Linguistically-Informed Low-Resource Chinese MT

Does encoding Chinese characters differently before feeding them to a neural MT model improve translation quality in low-resource settings?

This project tests six input representations — characters, morphemes, pinyin, IDS radicals, Cangjie, and Wubi — across two pretrained models (opus-mt-zh-en and NLLB-600M) fine-tuned with LoRA on 50–1000 examples.

## Representations

| Name | Description |
|------|-------------|
| `baseline` | Raw characters |
| `morphemes` | jieba word segmentation |
| `pinyin` | Tonal romanisation |
| `radicals` | CHISE IDS component decomposition |
| `cangjie` | Cangjie input method codes (kCangjie, UNIHAN) |
| `wubi` | Wubi86 input method codes (RIME dictionary) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install sacremoses sentencepiece accelerate unbabel-comet peft
pip install -r requirements-finetune.txt
```

Pre-cache HuggingFace assets on a login node before submitting:
```bash
python -c "from datasets import load_dataset; load_dataset('wmt19','zh-en',split='train')"
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
```

## Running

```bash
sbatch run_finetuning.sh          # submits a 3-task job array (one per seed)
python aggregate_results.py       # after all tasks finish
```

## Data

- `data/ids.txt` — CHISE IDS character decomposition database (~89K entries)
- `data/cangjie.txt` — UNIHAN kCangjie field (~29K entries, via unihan-etl)
- `data/wubi.txt` — Wubi86 dictionary (~71K entries, from RIME project)

## Key files

- `encoder.py` — `LinguisticEncoder` class (all representations)
- `finetune_experiment.py` — main experiment script
- `run_finetuning.sh` — SLURM job array submission script
- `aggregate_results.py` — combines per-seed CSVs into summary tables
