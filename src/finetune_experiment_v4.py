"""
finetune_experiment_v4.py — Selective decomposition experiment (Saunders 2020).

Key addition vs. v3:
  - 'selective_radicals': only decomposes OOV characters (not in model tokenizer vocab)
  - 'selective_morphemes': morpheme segment OOV chars, keep known chars intact
  - Same 4 standard reps for comparison

Tests the Saunders et al. (WAT 2020) hypothesis: "complete sub-character decomposition
often harms unseen character translation; decomposition before inference for unseen
characters only" gives better results.

Usage (SLURM array, task index → seed):
    PYTHONPATH=src python src/finetune_experiment_v4.py
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("SUBCHAR-MT v4 — SELECTIVE DECOMPOSITION (Saunders 2020)")
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

REPRESENTATIONS = ['baseline', 'morphemes', 'radicals', 'sentencepiece',
                   'selective_radicals', 'selective_morphemes']
TRAIN_SIZES     = [50, 250, 1000]
ALL_SEEDS       = [42, 123, 456, 789, 999]
MODELS          = ['opus-mt', 'nllb-600M']

MODEL_CONFIGS = {
    'opus-mt': {
        'model_id':     'Helsinki-NLP/opus-mt-zh-en',
        'family':       'marian',
        'label':        'opus-mt-zh-en (~74M)',
        'use_lora':     True,
        'lora_modules': ['q_proj', 'v_proj'],
    },
    'nllb-600M': {
        'model_id':     'facebook/nllb-200-distilled-600M',
        'family':       'nllb',
        'label':        'NLLB-600M (~600M)',
        'src_lang':     'zho_Hans',
        'tgt_lang':     'eng_Latn',
        'use_lora':     True,
        'lora_modules': ['q_proj', 'v_proj'],
    },
}

_DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_SP_MODEL     = os.path.join(_DATA_DIR, 'zh_sp.model')
_IDS_PATH     = os.path.join(_DATA_DIR, 'ids.txt')
_WUBI_PATH    = os.path.join(_DATA_DIR, 'wubi.txt')
_CANGJIE_PATH = os.path.join(_DATA_DIR, 'cangjie.txt')

_ENCODER = LinguisticEncoder(
    ids_path      = _IDS_PATH     if os.path.exists(_IDS_PATH)     else None,
    wubi_path     = _WUBI_PATH    if os.path.exists(_WUBI_PATH)    else None,
    cangjie_path  = _CANGJIE_PATH if os.path.exists(_CANGJIE_PATH) else None,
    sp_model_path = _SP_MODEL     if os.path.exists(_SP_MODEL)     else None,
)

# Per-model tokenizer vocabs for selective decomposition (loaded lazily)
_TOKENIZER_VOCABS: dict = {}


def get_tokenizer_vocab(model_key: str) -> set:
    if model_key not in _TOKENIZER_VOCABS:
        cfg = MODEL_CONFIGS[model_key]
        if cfg['family'] == 'marian':
            tok = MarianTokenizer.from_pretrained(cfg['model_id'])
        else:
            tok = AutoTokenizer.from_pretrained(cfg['model_id'],
                                                src_lang=cfg.get('src_lang', 'zho_Hans'))
        _TOKENIZER_VOCABS[model_key] = set(tok.get_vocab().keys())
    return _TOKENIZER_VOCABS[model_key]


# ── COMET ──────────────────────────────────────────────────────────────────

COMET_MODEL = None
try:
    from comet import download_model, load_from_checkpoint
    COMET_MODEL = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
    print("✓ COMET loaded\n")
except Exception as e:
    print(f"⚠ COMET unavailable: {e}\n")


# ── Helpers ────────────────────────────────────────────────────────────────

def apply_rep(texts, rep, model_key):
    if rep == 'baseline':     return texts
    if rep == 'morphemes':    return [_ENCODER.segment_morphemes(t) for t in texts]
    if rep == 'radicals':     return [_ENCODER.to_radicals(t) for t in texts]
    if rep == 'sentencepiece':return [_ENCODER.to_sentencepiece(t) for t in texts]
    if rep == 'selective_radicals':
        vocab = get_tokenizer_vocab(model_key)
        return [_ENCODER.to_selective_decomp(t, vocab, 'radicals') for t in texts]
    if rep == 'selective_morphemes':
        vocab = get_tokenizer_vocab(model_key)
        return [_ENCODER.to_selective_decomp(t, vocab, 'morphemes') for t in texts]
    return texts


def load_model_and_tokenizer(model_key):
    cfg = MODEL_CONFIGS[model_key]
    if cfg['family'] == 'marian':
        tok = MarianTokenizer.from_pretrained(cfg['model_id'])
        mdl = MarianMTModel.from_pretrained(cfg['model_id'])
    else:
        tok = AutoTokenizer.from_pretrained(cfg['model_id'], src_lang=cfg['src_lang'])
        mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_id'])
    return mdl, tok


def translate(model, tokenizer, source_texts, model_key, max_length=128):
    cfg = MODEL_CONFIGS[model_key]
    gen_kwargs = {'max_length': max_length}
    if cfg['family'] == 'nllb':
        gen_kwargs['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids(cfg['tgt_lang'])
    model.eval()
    preds = []
    for i in range(0, len(source_texts), 32):
        batch  = source_texts[i:i+32]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        preds.extend(tokenizer.decode(t, skip_special_tokens=True) for t in out)
    return preds


def evaluate(preds, refs, srcs=None):
    sent_bleus = [sentence_bleu(p, [r]).score for p, r in zip(preds, refs)]
    results = {
        'bleu': corpus_bleu(preds, [[r] for r in refs]).score,
        'bleu_std': float(np.std(sent_bleus)),
        'chrf': CHRF().corpus_score(preds, [[r] for r in refs]).score,
        'sent_bleus': sent_bleus,
    }
    if srcs and COMET_MODEL:
        try:
            comet_data = [{'src': s, 'mt': p, 'ref': r} for s, p, r in zip(srcs, preds, refs)]
            out = COMET_MODEL.predict(comet_data, batch_size=32, gpus=1)
            results['comet']        = float(np.mean(out.scores))
            results['comet_std']    = float(np.std(out.scores))
            results['comet_scores'] = [float(x) for x in out.scores]
        except Exception as e:
            print(f"  ⚠ COMET error: {e}")
            results['comet'] = results['comet_std'] = results['comet_scores'] = None
    else:
        results['comet'] = results['comet_std'] = results['comet_scores'] = None
    return results


def save_preds(path, sources, preds, refs, scores):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'sources': list(sources), 'predictions': preds, 'references': refs,
                   'sent_bleus': scores['sent_bleus'],
                   'comet_scores': scores.get('comet_scores')}, f, ensure_ascii=False)


def fine_tune(train_df, val_df, rep, model_key, out_dir, seed, epochs=5,
              batch_size=4, lr=5e-5):
    cfg = MODEL_CONFIGS[model_key]
    model, tokenizer = load_model_and_tokenizer(model_key)

    if cfg.get('use_lora') and _PEFT_AVAILABLE:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
            lora_dropout=0.1, target_modules=cfg['lora_modules'],
        )
        model = get_peft_model(model, lora_cfg)

    def make_hf_ds(df):
        data = [{'translation': {'zh': apply_rep([r['chinese']], rep, model_key)[0],
                                  'en': r['english']}}
                for _, r in df.iterrows()]
        return Dataset.from_list(data)

    def preprocess(examples):
        inps = [ex['zh'] for ex in examples['translation']]
        tgts = [ex['en'] for ex in examples['translation']]
        mi = tokenizer(inps, max_length=128, truncation=True, padding='max_length')
        lb = tokenizer(tgts, max_length=128, truncation=True, padding='max_length')
        mi['labels'] = lb['input_ids']
        return mi

    train_ds = make_hf_ds(train_df).map(preprocess, batched=True, remove_columns=['translation'])
    val_ds   = make_hf_ds(val_df).map(preprocess,   batched=True, remove_columns=['translation'])

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, eval_strategy='epoch', save_strategy='epoch',
        learning_rate=lr, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, num_train_epochs=epochs,
        weight_decay=0.01, save_total_limit=1, predict_with_generate=True,
        fp16=True, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        report_to='none', logging_steps=10, seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return model, tokenizer


def run(args):
    from datasets import load_dataset as _ld
    import warnings
    warnings.filterwarnings('ignore')

    print("Loading WMT19 zh-en...")
    ds = _ld('wmt19', 'zh-en', split='train')
    df = pd.DataFrame([{'chinese': ex['translation']['zh'],
                         'english': ex['translation']['en']}
                        for ex in ds.select(range(3000))])

    sh = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = sh.iloc[:500].copy()
    val_df   = sh.iloc[500:600].copy()
    remaining = sh.iloc[600:].copy().reset_index(drop=True)
    print(f"✓ {len(test_df)} test | {len(val_df)} val | {len(remaining)} pool\n")

    unseen_df = None
    if os.path.exists(args.unseen_csv):
        unseen_df = pd.read_csv(args.unseen_csv)
        print(f"✓ Unseen-char test: {len(unseen_df)} sentences\n")

    all_results = []

    for model_key in MODELS:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'='*80}\nMODEL: {cfg['label']}\n{'='*80}")

        for seed in args.seeds:
            train_pool = remaining.sample(n=max(TRAIN_SIZES), random_state=seed).reset_index(drop=True)

            for train_size in TRAIN_SIZES:
                train_df = train_pool.iloc[:train_size]

                for rep in REPRESENTATIONS:
                    out_dir = f"./models_v4/{model_key}_{rep}_{train_size}_s{seed}"
                    print(f"\n  {rep} | n={train_size} | seed={seed}")

                    model, tokenizer = fine_tune(train_df, val_df, rep, model_key, out_dir, seed)
                    model = model.cuda()

                    for test_set, eval_df in [('regular', test_df),
                                               ('unseen', unseen_df)]:
                        if eval_df is None:
                            continue
                        src = apply_rep(eval_df['chinese'].tolist(), rep, model_key)
                        preds = translate(model, tokenizer, src, model_key)
                        scores = evaluate(preds, eval_df['english'].tolist(),
                                          srcs=eval_df['chinese'].tolist())

                        bleu_s = f"BLEU={scores['bleu']:.2f}"
                        comet_s = f"COMET={scores['comet']:.4f}" if scores['comet'] else ""
                        print(f"    [{test_set:7s}] {bleu_s}  chrF={scores['chrf']:.2f}  {comet_s}")

                        if args.save_predictions:
                            suffix = '_unseen' if test_set == 'unseen' else ''
                            save_preds(
                                f"{args.predictions_dir}/{model_key}_{rep}_{train_size}_s{seed}{suffix}_preds.json",
                                eval_df['chinese'].tolist(), preds,
                                eval_df['english'].tolist(), scores,
                            )

                        all_results.append(dict(
                            condition='finetuned', model=model_key, representation=rep,
                            train_size=train_size, seed=seed, test_set=test_set,
                            bleu=scores['bleu'], bleu_std=scores['bleu_std'],
                            chrf=scores['chrf'], comet=scores['comet'],
                        ))

                    pd.DataFrame(all_results).to_csv(
                        f'v4_results_intermediate{args.suffix}.csv', index=False)
                    del model, tokenizer
                    torch.cuda.empty_cache()
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(f'v4_results_FINAL{args.suffix}.csv', index=False)
    print(f"\n✓ Saved v4_results_FINAL{args.suffix}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unseen_csv',       default='data/unseen_char_test.csv')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--predictions_dir',  default='results/predictions_v4')
    cli = parser.parse_args()

    task = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    if task >= 0:
        cli.seeds  = [ALL_SEEDS[task]]
        cli.suffix = f'_s{cli.seeds[0]}'
        print(f'Array task {task} → seed {cli.seeds[0]}\n')
    else:
        cli.seeds  = ALL_SEEDS
        cli.suffix = ''

    run(cli)
