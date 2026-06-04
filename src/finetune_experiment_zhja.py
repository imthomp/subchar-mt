"""
finetune_experiment_zhja.py — CJK transfer: Zh→Ja with shared radicals.

Tests whether sub-character representations (radicals) specifically help
Chinese→Japanese translation, where shared CJK ideographs create a
cross-lingual bridge. Hypothesis: radical decomposition should help more
for zh→ja than zh→en because Japanese kanji share radical structure with Chinese.

Data: FLORES+ cmn_Hans / jpn_Jpan (997 train, 1012 test — stored in data/ CSVs).
Models: opus-mt-zh-ja (Helsinki-NLP), NLLB-200-distilled-600M.
Reps: baseline, morphemes, radicals, sentencepiece (same 4 as v3).
Train sizes: 50, 250, 500 (capped by FLORES+ train set size).
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
print("SUBCHAR-MT Zh→Ja CJK EXTENSION")
print("=" * 80)

import torch
from datasets import Dataset
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

REPRESENTATIONS = ['baseline', 'morphemes', 'radicals', 'sentencepiece']
TRAIN_SIZES     = [50, 250, 500]
ALL_SEEDS       = [42, 123, 456, 789, 999]

MODEL_CONFIGS = {
    'opus-mt-zhja': {
        'model_id':     'Helsinki-NLP/opus-mt-tc-big-zh-ja',
        'family':       'marian',
        'label':        'opus-mt-tc-big-zh-ja',
        'use_lora':     True,
        'lora_modules': ['q_proj', 'v_proj'],
        'tgt_lang':     None,
    },
    'nllb-600M': {
        'model_id':     'facebook/nllb-200-distilled-600M',
        'family':       'nllb',
        'label':        'NLLB-600M (~600M)',
        'src_lang':     'zho_Hans',
        'tgt_lang':     'jpn_Jpan',
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


def translate(model, tokenizer, src_texts, model_key, max_length=256):
    cfg = MODEL_CONFIGS[model_key]
    gen_kwargs = {'max_length': max_length}
    if cfg['family'] == 'nllb':
        gen_kwargs['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids(cfg['tgt_lang'])
    model.eval()
    preds = []
    for i in range(0, len(src_texts), 16):
        batch  = src_texts[i:i+16]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=256)
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


def fine_tune(train_df, val_df, rep, model_key, out_dir, seed):
    cfg = MODEL_CONFIGS[model_key]
    model, tok = load_model_tok(model_key)
    if cfg.get('use_lora') and _PEFT_AVAILABLE:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
            lora_dropout=0.1, target_modules=cfg['lora_modules'],
        ))

    def make_ds(df):
        src_col, tgt_col = 'chinese', 'japanese'
        data = [{'src': apply_rep([r[src_col]], rep)[0], 'tgt': r[tgt_col]}
                for _, r in df.iterrows()]
        return Dataset.from_list(data)

    def preprocess(examples):
        mi = tok(examples['src'], max_length=256, truncation=True, padding='max_length')
        lb = tok(examples['tgt'], max_length=256, truncation=True, padding='max_length')
        mi['labels'] = lb['input_ids']
        return mi

    train_ds = make_ds(train_df).map(preprocess, batched=True, remove_columns=['src','tgt'])
    val_ds   = make_ds(val_df).map(preprocess,   batched=True, remove_columns=['src','tgt'])

    tr_args = Seq2SeqTrainingArguments(
        output_dir=out_dir, eval_strategy='epoch', save_strategy='epoch',
        learning_rate=5e-5, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        num_train_epochs=5, weight_decay=0.01, save_total_limit=1,
        predict_with_generate=True, fp16=True, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', report_to='none', logging_steps=10, seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=tr_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    return model, tok


def run(args):
    train_csv = os.path.join(_DATA_DIR, 'flores_zhja_train.csv')
    test_csv  = os.path.join(_DATA_DIR, 'flores_zhja_test.csv')
    assert os.path.exists(train_csv), f"Missing: {train_csv}  (run on login node first)"
    assert os.path.exists(test_csv),  f"Missing: {test_csv}"

    full_train = pd.read_csv(train_csv)
    test_df    = pd.read_csv(test_csv)
    val_df     = full_train.sample(n=min(100, len(full_train)), random_state=42)
    pool       = full_train.drop(val_df.index).reset_index(drop=True)
    print(f"✓ FLORES+ zh-ja: {len(pool)} train pool | {len(val_df)} val | {len(test_df)} test\n")

    all_results = []

    for model_key in ['opus-mt-zhja', 'nllb-600M']:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'='*80}\nMODEL: {cfg['label']}\n{'='*80}")

        for seed in args.seeds:
            train_pool = pool.sample(n=min(max(TRAIN_SIZES), len(pool)),
                                     random_state=seed).reset_index(drop=True)
            for train_size in TRAIN_SIZES:
                if train_size > len(train_pool):
                    continue
                train_df = train_pool.iloc[:train_size]
                for rep in REPRESENTATIONS:
                    out_dir = f"./models_zhja/{model_key}_{rep}_{train_size}_s{seed}"
                    print(f"\n  {rep} | n={train_size} | seed={seed}")

                    model, tok = fine_tune(train_df, val_df, rep, model_key, out_dir, seed)
                    model = model.cuda()

                    src = apply_rep(test_df['chinese'].tolist(), rep)
                    preds  = translate(model, tok, src, model_key)
                    scores = evaluate(preds, test_df['japanese'].tolist(),
                                      srcs=test_df['chinese'].tolist())

                    comet_s = f"COMET={scores['comet']:.4f}" if scores['comet'] else ""
                    print(f"    BLEU={scores['bleu']:.2f}  chrF={scores['chrf']:.2f}  {comet_s}")

                    if args.save_predictions:
                        save_preds(
                            f"{args.preds_dir}/{model_key}_{rep}_{train_size}_s{seed}_preds.json",
                            test_df['chinese'].tolist(), preds,
                            test_df['japanese'].tolist(), scores,
                        )

                    all_results.append(dict(
                        model=model_key, representation=rep,
                        train_size=train_size, seed=seed,
                        bleu=scores['bleu'], bleu_std=scores['bleu_std'],
                        chrf=scores['chrf'], comet=scores['comet'],
                    ))
                    pd.DataFrame(all_results).to_csv(
                        f'zhja_results_intermediate{args.suffix}.csv', index=False)
                    del model, tok
                    torch.cuda.empty_cache()
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)

    df = pd.DataFrame(all_results)
    df.to_csv(f'zhja_results_FINAL{args.suffix}.csv', index=False)
    print(f"\n✓ Saved zhja_results_FINAL{args.suffix}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--preds_dir', default='results/predictions_zhja')
    cli = parser.parse_args()

    task = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    if task >= 0:
        cli.seeds  = [ALL_SEEDS[task]]
        cli.suffix = f'_s{ALL_SEEDS[task]}'
    else:
        cli.seeds  = ALL_SEEDS
        cli.suffix = ''

    run(cli)
