"""
finetune_experiment_v5.py — Decomposition-depth ablation (Han et al. 2512.15556).

Han, Jones & Smeaton found that intermediate-depth IDS decomposition (rxd2) collapses
performance while ideograph-level (rxd1) and near-stroke (rxd3) roughly match the
word baseline. Our current 'radicals' condition uses CHISE IDS at depth-1.

This experiment adds:
  - radicals_d1: depth-1 (current — direct IDS components only)
  - radicals_d2: depth-2 (intermediate — Han et al.'s "collapse" condition)
  - radicals_full: recursive to primitives (near-stroke level)

Tests whether our radical results are confounded by decomposition depth.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print("=" * 80)
print("SUBCHAR-MT v5 — DECOMPOSITION DEPTH ABLATION")
print("=" * 80)

import torch
from datasets import load_dataset, Dataset
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from sacrebleu import corpus_bleu, sentence_bleu
from sacrebleu.metrics import CHRF
from encoder import LinguisticEncoder

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

print(f"✓ PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# ── Config ─────────────────────────────────────────────────────────────────

REPRESENTATIONS = ['baseline', 'radicals_d1', 'radicals_d2', 'radicals_full']
TRAIN_SIZES     = [50, 250, 1000]
ALL_SEEDS       = [42, 123, 456, 789, 999]
MODELS          = ['opus-mt', 'nllb-600M']

MODEL_CONFIGS = {
    'opus-mt': {
        'model_id': 'Helsinki-NLP/opus-mt-zh-en', 'family': 'marian',
        'label': 'opus-mt-zh-en', 'use_lora': True, 'lora_modules': ['q_proj', 'v_proj'],
    },
    'nllb-600M': {
        'model_id': 'facebook/nllb-200-distilled-600M', 'family': 'nllb',
        'label': 'NLLB-600M', 'src_lang': 'zho_Hans', 'tgt_lang': 'eng_Latn',
        'use_lora': True, 'lora_modules': ['q_proj', 'v_proj'],
    },
}

_DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_IDS_PATH     = os.path.join(_DATA_DIR, 'ids.txt')
_SP_MODEL     = os.path.join(_DATA_DIR, 'zh_sp.model')

_ENCODER = LinguisticEncoder(
    ids_path      = _IDS_PATH  if os.path.exists(_IDS_PATH)  else None,
    sp_model_path = _SP_MODEL  if os.path.exists(_SP_MODEL)  else None,
)

COMET_MODEL = None
try:
    from comet import download_model, load_from_checkpoint
    COMET_MODEL = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
    print("✓ COMET loaded\n")
except Exception as e:
    print(f"⚠ COMET unavailable: {e}\n")


def apply_rep(texts, rep):
    if rep == 'baseline':      return texts
    if rep == 'radicals_d1':   return [_ENCODER.to_radicals(t, depth=1) for t in texts]
    if rep == 'radicals_d2':   return [_ENCODER.to_radicals(t, depth=2) for t in texts]
    if rep == 'radicals_full': return [_ENCODER.to_radicals(t, depth=6) for t in texts]
    return texts


def load_model_tok(model_key):
    cfg = MODEL_CONFIGS[model_key]
    if cfg['family'] == 'marian':
        tok = MarianTokenizer.from_pretrained(cfg['model_id'])
        mdl = MarianMTModel.from_pretrained(cfg['model_id'])
    else:
        tok = AutoTokenizer.from_pretrained(cfg['model_id'], src_lang=cfg['src_lang'])
        mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_id'])
    return mdl, tok


def translate(model, tok, srcs, model_key, max_length=128):
    cfg = MODEL_CONFIGS[model_key]
    gen_kw = {'max_length': max_length}
    if cfg['family'] == 'nllb':
        gen_kw['forced_bos_token_id'] = tok.convert_tokens_to_ids(cfg['tgt_lang'])
    model.eval()
    preds = []
    for i in range(0, len(srcs), 32):
        batch  = srcs[i:i+32]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        preds.extend(tok.decode(t, skip_special_tokens=True) for t in out)
    return preds


def evaluate(preds, refs, srcs=None):
    sent_bleus = [sentence_bleu(p, [r]).score for p, r in zip(preds, refs)]
    res = {'bleu': corpus_bleu(preds, [[r] for r in refs]).score,
           'bleu_std': float(np.std(sent_bleus)),
           'chrf': CHRF().corpus_score(preds, [[r] for r in refs]).score,
           'sent_bleus': sent_bleus}
    if srcs and COMET_MODEL:
        try:
            out = COMET_MODEL.predict(
                [{'src': s, 'mt': p, 'ref': r} for s, p, r in zip(srcs, preds, refs)],
                batch_size=32, gpus=1)
            res['comet'] = float(np.mean(out.scores))
            res['comet_scores'] = [float(x) for x in out.scores]
        except Exception as e:
            res['comet'] = res['comet_scores'] = None
    else:
        res['comet'] = res['comet_scores'] = None
    return res


def fine_tune(train_df, val_df, rep, model_key, out_dir, seed):
    cfg = MODEL_CONFIGS[model_key]
    model, tok = load_model_tok(model_key)
    if cfg.get('use_lora') and _PEFT_AVAILABLE:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
            lora_dropout=0.1, target_modules=cfg['lora_modules']))

    def make_ds(df):
        return Dataset.from_list([
            {'src': apply_rep([r['chinese']], rep)[0], 'tgt': r['english']}
            for _, r in df.iterrows()])

    def preprocess(examples):
        mi = tok(examples['src'], max_length=128, truncation=True, padding='max_length')
        lb = tok(examples['tgt'], max_length=128, truncation=True, padding='max_length')
        mi['labels'] = lb['input_ids']
        return mi

    tr = make_ds(train_df).map(preprocess, batched=True, remove_columns=['src', 'tgt'])
    vl = make_ds(val_df).map(preprocess,   batched=True, remove_columns=['src', 'tgt'])

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=out_dir, eval_strategy='epoch', save_strategy='epoch',
            learning_rate=5e-5, per_device_train_batch_size=4,
            per_device_eval_batch_size=4, num_train_epochs=5, weight_decay=0.01,
            save_total_limit=1, predict_with_generate=True, fp16=True,
            load_best_model_at_end=True, metric_for_best_model='eval_loss',
            report_to='none', logging_steps=10, seed=seed,
        ),
        train_dataset=tr, eval_dataset=vl,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    return model, tok


def run(args):
    print("Loading WMT19 zh-en...")
    ds = load_dataset('wmt19', 'zh-en', split='train')
    df = pd.DataFrame([{'chinese': ex['translation']['zh'],
                         'english': ex['translation']['en']}
                        for ex in ds.select(range(3000))])
    sh = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df   = sh.iloc[:500].copy()
    val_df    = sh.iloc[500:600].copy()
    remaining = sh.iloc[600:].copy().reset_index(drop=True)
    print(f"✓ {len(test_df)} test | {len(val_df)} val | {len(remaining)} pool\n")

    all_results = []

    for model_key in MODELS:
        print(f"\n{'='*70}\nMODEL: {MODEL_CONFIGS[model_key]['label']}\n{'='*70}")
        for seed in args.seeds:
            train_pool = remaining.sample(n=max(TRAIN_SIZES), random_state=seed).reset_index(drop=True)
            for train_size in TRAIN_SIZES:
                train_df = train_pool.iloc[:train_size]
                for rep in REPRESENTATIONS:
                    out_dir = f"./models_v5/{model_key}_{rep}_{train_size}_s{seed}"
                    print(f"\n  {rep} | n={train_size} | seed={seed}")

                    model, tok = fine_tune(train_df, val_df, rep, model_key, out_dir, seed)
                    model = model.cuda()

                    src    = apply_rep(test_df['chinese'].tolist(), rep)
                    preds  = translate(model, tok, src, model_key)
                    scores = evaluate(preds, test_df['english'].tolist(),
                                      srcs=test_df['chinese'].tolist())
                    print(f"    BLEU={scores['bleu']:.2f}  chrF={scores['chrf']:.2f}"
                          + (f"  COMET={scores['comet']:.4f}" if scores['comet'] else ""))

                    all_results.append(dict(
                        model=model_key, representation=rep,
                        train_size=train_size, seed=seed,
                        bleu=scores['bleu'], bleu_std=scores['bleu_std'],
                        chrf=scores['chrf'], comet=scores['comet'],
                    ))
                    pd.DataFrame(all_results).to_csv(
                        f'v5_results_intermediate{args.suffix}.csv', index=False)
                    del model, tok
                    torch.cuda.empty_cache()
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(f'v5_results_FINAL{args.suffix}.csv', index=False)
    print(f"\n✓ Saved v5_results_FINAL{args.suffix}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    cli = parser.parse_args()
    task = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    if task >= 0:
        cli.seeds  = [ALL_SEEDS[task]]
        cli.suffix = f'_s{ALL_SEEDS[task]}'
        print(f'Array task {task} → seed {cli.seeds[0]}\n')
    else:
        cli.seeds  = ALL_SEEDS
        cli.suffix = ''
    run(cli)
