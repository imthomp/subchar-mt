"""
finetune_experiment_v3.py — Targeted diagnostic run.

Differences from v2:
  - Subset of representations: baseline, morphemes, radicals, sentencepiece
  - Subset of train sizes: 50, 250, 1000
  - Evaluates on BOTH regular test set and unseen-character test set
  - Extracts encoder hidden states after each condition (for probing)
  - Keeps predictions for both test sets

Usage (SLURM array, task index → seed):
    PYTHONPATH=src python src/finetune_experiment_v3.py

CLI flags:
    --unseen_csv        path to unseen-character test CSV (default: data/unseen_char_test.csv)
    --save_predictions  save prediction JSONs
    --extract_states    save encoder hidden states as .npz
    --states_dir        output dir for .npz files (default: results/probing/)
    --predictions_dir   output dir for prediction JSONs (default: results/predictions_v3/)
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
print("SUBCHAR-MT v3 — DIAGNOSTIC RUN (unseen-char + probing)")
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
print()

# ── Config ─────────────────────────────────────────────────────────────────

REPRESENTATIONS = ['baseline', 'morphemes', 'radicals', 'sentencepiece']
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

_DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_SP_MODEL    = os.path.join(_DATA_DIR, 'zh_sp.model')
_IDS_PATH    = os.path.join(_DATA_DIR, 'ids.txt')
_WUBI_PATH   = os.path.join(_DATA_DIR, 'wubi.txt')
_CANGJIE_PATH = os.path.join(_DATA_DIR, 'cangjie.txt')

_ENCODER = LinguisticEncoder(
    ids_path      = _IDS_PATH     if os.path.exists(_IDS_PATH)     else None,
    wubi_path     = _WUBI_PATH    if os.path.exists(_WUBI_PATH)    else None,
    cangjie_path  = _CANGJIE_PATH if os.path.exists(_CANGJIE_PATH) else None,
    sp_model_path = _SP_MODEL     if os.path.exists(_SP_MODEL)     else None,
)

# ── COMET ──────────────────────────────────────────────────────────────────

COMET_MODEL = None
try:
    from comet import download_model, load_from_checkpoint
    COMET_MODEL = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
    print("✓ COMET loaded\n")
except Exception as e:
    print(f"⚠ COMET unavailable: {e}\n")

# ── Helpers ────────────────────────────────────────────────────────────────

def apply_rep(texts, rep):
    if rep == 'baseline':     return texts
    if rep == 'morphemes':    return [_ENCODER.segment_morphemes(t) for t in texts]
    if rep == 'radicals':     return [_ENCODER.to_radicals(t) for t in texts]
    if rep == 'sentencepiece':return [_ENCODER.to_sentencepiece(t) for t in texts]
    if rep == 'pinyin':       return [_ENCODER.to_pinyin(t) for t in texts]
    if rep == 'byte':         return [_ENCODER.to_bytes(t) for t in texts]
    if rep == 'random_index': return [_ENCODER.to_random_index(t) for t in texts]
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
        'bleu':     corpus_bleu(preds, [[r] for r in refs]).score,
        'bleu_std': float(np.std(sent_bleus)),
        'chrf':     CHRF().corpus_score(preds, [[r] for r in refs]).score,
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


def save_predictions(path, sources, preds, refs, scores):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'sources': list(sources), 'predictions': preds, 'references': refs,
            'sent_bleus': scores['sent_bleus'],
            'comet_scores': scores.get('comet_scores'),
        }, f, ensure_ascii=False)


def extract_hidden_states(model, tokenizer, texts, layer=-1, pool='mean',
                          batch_size=16, max_length=128):
    """Return mean-pooled encoder hidden states, shape [n, d]."""
    model.eval()
    all_states = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            # Use the base model's encoder (works for both Marian and NLLB)
            base = model.model if hasattr(model, 'model') else model
            encoder = base.encoder if hasattr(base, 'encoder') else base.get_encoder()
            hs_out = encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
            )
        hs   = hs_out.hidden_states[layer]          # [batch, seq, d]
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1)   # mean pool over non-pad tokens
        all_states.append(pooled.cpu().float().numpy())
    return np.vstack(all_states)


# ── Data loading ───────────────────────────────────────────────────────────

def load_main_data(max_samples=3000):
    print("Loading WMT19 zh-en...")
    ds = load_dataset('wmt19', 'zh-en', split='train')
    ds = ds.select(range(min(max_samples, len(ds))))
    df = pd.DataFrame([{'chinese': ex['translation']['zh'],
                         'english': ex['translation']['en']} for ex in ds])
    print(f"✓ {len(df)} examples\n")
    return df


def make_fixed_splits(df, val_size=100, test_size=500, seed=42):
    sh = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = sh.iloc[:test_size].copy()
    val_df   = sh.iloc[test_size:test_size+val_size].copy()
    remaining = sh.iloc[test_size+val_size:].copy().reset_index(drop=True)
    print(f"✓ {len(test_df)} test | {len(val_df)} val | {len(remaining)} pool\n")
    return test_df, val_df, remaining


def load_unseen_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠ Unseen-char test set not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    print(f"✓ Unseen-char test set: {len(df)} sentences from {csv_path}\n")
    return df


# ── Fine-tuning ────────────────────────────────────────────────────────────

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
        model.print_trainable_parameters()

    def make_hf_ds(df):
        data = [{'translation': {'zh': apply_rep([r['chinese']], rep)[0], 'en': r['english']}}
                for _, r in df.iterrows()]
        return Dataset.from_list(data)

    def preprocess(examples):
        inps  = [ex['zh'] for ex in examples['translation']]
        tgts  = [ex['en'] for ex in examples['translation']]
        mi = tokenizer(inps,  max_length=128, truncation=True, padding='max_length')
        lb = tokenizer(tgts,  max_length=128, truncation=True, padding='max_length')
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


# ── Main experiment ────────────────────────────────────────────────────────

def run(args):
    df_full = load_main_data()
    test_df, val_df, remaining = make_fixed_splits(df_full)
    unseen_df = load_unseen_data(args.unseen_csv)

    all_results = []

    for model_key in MODELS:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'=' * 80}")
        print(f"MODEL: {cfg['label']}")
        print(f"{'=' * 80}")

        for seed in args.seeds:
            train_pool = remaining.sample(n=max(TRAIN_SIZES), random_state=seed).reset_index(drop=True)

            for train_size in TRAIN_SIZES:
                train_df = train_pool.iloc[:train_size]

                for rep in REPRESENTATIONS:
                    out_dir = f"./models_v3/{model_key}_{rep}_{train_size}_s{seed}"
                    print(f"\n  [{cfg['label']}] {rep} | n={train_size} | seed={seed}")

                    model, tokenizer = fine_tune(
                        train_df, val_df, rep, model_key, out_dir, seed
                    )
                    model = model.cuda()

                    # ── Regular test set ─────────────────────────────────
                    src_texts = apply_rep(test_df['chinese'].tolist(), rep)
                    preds     = translate(model, tokenizer, src_texts, model_key)
                    scores    = evaluate(preds, test_df['english'].tolist(),
                                         srcs=test_df['chinese'].tolist())
                    print(f"    [test]   BLEU={scores['bleu']:.2f}  "
                          f"chrF={scores['chrf']:.2f}  "
                          f"COMET={scores['comet']:.4f}" if scores['comet'] else
                          f"    [test]   BLEU={scores['bleu']:.2f}  chrF={scores['chrf']:.2f}")

                    if args.save_predictions:
                        save_predictions(
                            f"{args.predictions_dir}/{model_key}_{rep}_{train_size}_s{seed}_preds.json",
                            test_df['chinese'].tolist(), preds,
                            test_df['english'].tolist(), scores,
                        )

                    row = dict(condition='finetuned', model=model_key, representation=rep,
                               train_size=train_size, seed=seed, test_set='regular',
                               bleu=scores['bleu'], bleu_std=scores['bleu_std'],
                               chrf=scores['chrf'], comet=scores['comet'],
                               comet_std=scores.get('comet_std'))
                    all_results.append(row)

                    # ── Unseen-character test set ─────────────────────────
                    if unseen_df is not None:
                        usrc = apply_rep(unseen_df['chinese'].tolist(), rep)
                        upreds = translate(model, tokenizer, usrc, model_key)
                        uscores = evaluate(upreds, unseen_df['english'].tolist(),
                                           srcs=unseen_df['chinese'].tolist())
                        print(f"    [unseen] BLEU={uscores['bleu']:.2f}  "
                              f"chrF={uscores['chrf']:.2f}  "
                              f"COMET={uscores['comet']:.4f}" if uscores['comet'] else
                              f"    [unseen] BLEU={uscores['bleu']:.2f}  chrF={uscores['chrf']:.2f}")

                        if args.save_predictions:
                            save_predictions(
                                f"{args.predictions_dir}/{model_key}_{rep}_{train_size}_s{seed}_unseen_preds.json",
                                unseen_df['chinese'].tolist(), upreds,
                                unseen_df['english'].tolist(), uscores,
                            )

                        urow = dict(condition='finetuned', model=model_key, representation=rep,
                                    train_size=train_size, seed=seed, test_set='unseen',
                                    bleu=uscores['bleu'], bleu_std=uscores['bleu_std'],
                                    chrf=uscores['chrf'], comet=uscores['comet'],
                                    comet_std=uscores.get('comet_std'))
                        all_results.append(urow)

                    # ── Hidden-state extraction ───────────────────────────
                    if args.extract_states:
                        print(f"    [states] extracting encoder hidden states...")
                        src_texts_base = test_df['chinese'].tolist()  # always use original Chinese
                        enc_texts = apply_rep(src_texts_base, rep)
                        states = extract_hidden_states(model, tokenizer, enc_texts)
                        states_path = (f"{args.states_dir}/"
                                       f"{model_key}_{rep}_{train_size}_s{seed}_states.npz")
                        pathlib.Path(args.states_dir).mkdir(parents=True, exist_ok=True)
                        np.savez(states_path,
                                 states=states,
                                 sources=np.array(src_texts_base, dtype=object))
                        print(f"    [states] saved {states.shape} → {states_path}")

                    # ── Save intermediate results + cleanup ──────────────
                    pd.DataFrame(all_results).to_csv(
                        f'v3_results_intermediate{args.suffix}.csv', index=False
                    )
                    del model, tokenizer
                    torch.cuda.empty_cache()
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'v3_results_FINAL{args.suffix}.csv', index=False)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'#' * 80}")
    print("SUMMARY: regular test set")
    print(f"{'#' * 80}")
    reg = results_df[results_df['test_set'] == 'regular'].copy()
    reg['comet'] = pd.to_numeric(reg['comet'], errors='coerce')
    summ = reg.groupby(['model', 'representation', 'train_size']).agg(
        bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
        chrf_mean=('chrf', 'mean'), comet_mean=('comet', 'mean'),
    ).round(3)
    print(summ.to_string())

    if unseen_df is not None:
        print(f"\n{'#' * 80}")
        print("SUMMARY: unseen-character test set")
        print(f"{'#' * 80}")
        uns = results_df[results_df['test_set'] == 'unseen'].copy()
        uns['comet'] = pd.to_numeric(uns['comet'], errors='coerce')
        summ_u = uns.groupby(['model', 'representation', 'train_size']).agg(
            bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
            chrf_mean=('chrf', 'mean'), comet_mean=('comet', 'mean'),
        ).round(3)
        print(summ_u.to_string())

    print(f"\n✓ Results saved to v3_results_FINAL{args.suffix}.csv")
    return results_df


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unseen_csv',       default='data/unseen_char_test.csv')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--extract_states',   action='store_true')
    parser.add_argument('--predictions_dir',  default='results/predictions_v3')
    parser.add_argument('--states_dir',       default='results/probing')
    cli = parser.parse_args()

    array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    if array_task_id >= 0:
        cli.seeds  = [ALL_SEEDS[array_task_id]]
        cli.suffix = f'_s{cli.seeds[0]}'
        print(f'Array task {array_task_id} → seed {cli.seeds[0]}\n')
    else:
        cli.seeds  = ALL_SEEDS
        cli.suffix = ''
        print('Standalone mode: all seeds\n')

    run(cli)
