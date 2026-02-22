"""
Fine-tuning experiment for linguistically-informed low-resource MT
BYU Supercomputer submission script

Author: Isaac (PhD student)
Course: CS 501R
Date: 2026-01-30
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("LINGUISTICALLY-INFORMED LOW-RESOURCE MT EXPERIMENT")
print("=" * 80)
print(f"Running on: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'}")
print(f"Working directory: {os.getcwd()}\n")

import torch
from datasets import load_dataset, Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
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
    print("Warning: peft not installed — LoRA disabled")

print(f"✓ PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
print()

# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_CONFIGS = {
    'opus-mt': {
        'model_id':       'Helsinki-NLP/opus-mt-zh-en',
        'family':         'marian',
        'label':          'opus-mt-zh-en (~74M)',
        'use_lora':       True,
        'lora_modules':   ['q_proj', 'v_proj'],
    },
    'nllb-600M': {
        'model_id':       'facebook/nllb-200-distilled-600M',
        'family':         'nllb',
        'label':          'NLLB-600M (~600M)',
        'src_lang':       'zho_Hans',
        'tgt_lang':       'eng_Latn',
        'use_lora':       True,
        'lora_modules':   ['q_proj', 'v_proj'],
    },
}


def load_model_and_tokenizer(model_key):
    cfg = MODEL_CONFIGS[model_key]
    if cfg['family'] == 'marian':
        tokenizer = MarianTokenizer.from_pretrained(cfg['model_id'])
        model     = MarianMTModel.from_pretrained(cfg['model_id'])
    elif cfg['family'] == 'nllb':
        tokenizer = AutoTokenizer.from_pretrained(
            cfg['model_id'], src_lang=cfg['src_lang']
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_id'])
    return model, tokenizer


# ============================================================================
# COMET — load once at startup
# ============================================================================

COMET_MODEL = None
try:
    from comet import download_model, load_from_checkpoint
    comet_path  = download_model('Unbabel/wmt22-comet-da')
    COMET_MODEL = load_from_checkpoint(comet_path)
    print("✓ COMET model loaded\n")
except Exception as e:
    print(f"Warning: COMET not available: {e}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_IDS_PATH     = os.path.join(_DATA_DIR, 'ids.txt')
_WUBI_PATH    = os.path.join(_DATA_DIR, 'wubi.txt')
_CANGJIE_PATH = os.path.join(_DATA_DIR, 'cangjie.txt')


def _make_encoder():
    return LinguisticEncoder(
        ids_path     = _IDS_PATH     if os.path.exists(_IDS_PATH)     else None,
        wubi_path    = _WUBI_PATH    if os.path.exists(_WUBI_PATH)    else None,
        cangjie_path = _CANGJIE_PATH if os.path.exists(_CANGJIE_PATH) else None,
    )


def load_and_prepare_data(max_samples=3000):
    """Load WMT19 zh-en and precompute all linguistic representations."""
    print(f"Loading WMT19 zh-en ({max_samples} samples)...")
    dataset = load_dataset('wmt19', 'zh-en', split='train')
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    df = pd.DataFrame(dataset)
    df['chinese'] = df['translation'].apply(lambda x: x['zh'])
    df['english'] = df['translation'].apply(lambda x: x['en'])

    print("Creating linguistic representations...")
    enc = _make_encoder()
    df['pinyin']        = df['chinese'].apply(enc.to_pinyin)
    df['pinyin_no_tone']= df['chinese'].apply(lambda x: enc.to_pinyin(x, with_tone=False))
    df['morphemes']     = df['chinese'].apply(enc.segment_morphemes)
    df['radicals']      = df['chinese'].apply(enc.to_radicals)
    df['wubi']          = df['chinese'].apply(lambda x: ' '.join(enc.to_wubi(x)))
    df['cangjie']       = df['chinese'].apply(lambda x: ' '.join(enc.to_cangjie(x)))

    print(f"✓ {len(df)} examples prepared\n")
    return df


def create_fixed_eval_sets(df, val_size=100, test_size=500, seed=42):
    """Hold-out a fixed test and val set shared across all seeds.

    Only the training sample drawn from `remaining` varies per seed,
    ensuring all conditions are evaluated on identical held-out data.
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df   = df_shuffled.iloc[:test_size].copy()
    val_df    = df_shuffled.iloc[test_size:test_size + val_size].copy()
    remaining = df_shuffled.iloc[test_size + val_size:].copy().reset_index(drop=True)
    print(f"✓ {len(test_df)} test | {len(val_df)} val | {len(remaining)} for training\n")
    return test_df, val_df, remaining


def sample_train_data(remaining_df, train_size, seed):
    return remaining_df.sample(
        n=min(train_size, len(remaining_df)), random_state=seed
    ).reset_index(drop=True)


# Load data once
df_full = load_and_prepare_data(max_samples=3000)
test_data, val_data, remaining_data = create_fixed_eval_sets(df_full)

# ============================================================================
# REPRESENTATIONS
# ============================================================================

_ENCODER = _make_encoder()


def apply_representation(texts, representation):
    """Map a list of Chinese strings to the target representation."""
    if representation == 'baseline':
        return texts
    elif representation == 'pinyin':
        return [_ENCODER.to_pinyin(t) for t in texts]
    elif representation == 'pinyin_no_tone':
        return [_ENCODER.to_pinyin(t, with_tone=False) for t in texts]
    elif representation == 'morphemes':
        return [_ENCODER.segment_morphemes(t) for t in texts]
    elif representation == 'radicals':
        return [_ENCODER.to_radicals(t) for t in texts]
    elif representation == 'wubi':
        return [' '.join(_ENCODER.to_wubi(t)) for t in texts]
    elif representation == 'cangjie':
        return [' '.join(_ENCODER.to_cangjie(t)) for t in texts]
    return texts


# ============================================================================
# INFERENCE
# ============================================================================

def translate_with_model(model, tokenizer, source_texts, model_key, max_length=128):
    """Batch-translate source_texts using the given model."""
    cfg = MODEL_CONFIGS[model_key]
    generate_kwargs = {'max_length': max_length}

    # NLLB requires forcing the target-language BOS token
    if cfg['family'] == 'nllb':
        generate_kwargs['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids(
            cfg['tgt_lang']
        )

    predictions = []
    model.eval()
    for i in range(0, len(source_texts), 32):
        batch  = source_texts[i:i + 32]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **generate_kwargs)
        predictions.extend(
            [tokenizer.decode(t, skip_special_tokens=True) for t in out]
        )
    return predictions


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_translations(predictions, references, sources=None):
    """Compute BLEU, chrF, and COMET."""
    results = {}

    # BLEU
    sent_bleus = [sentence_bleu(p, [r]).score for p, r in zip(predictions, references)]
    results['bleu']     = corpus_bleu(predictions, [[r] for r in references]).score
    results['bleu_std'] = float(np.std(sent_bleus))

    # chrF — character n-gram F-score, better for low-resource + morphologically rich
    results['chrf'] = CHRF().corpus_score(predictions, [[r] for r in references]).score

    # COMET — best correlation with human judgement
    if sources is not None and COMET_MODEL is not None:
        try:
            comet_data = [{'src': s, 'mt': p, 'ref': r}
                          for s, p, r in zip(sources, predictions, references)]
            comet_out  = COMET_MODEL.predict(comet_data, batch_size=32, gpus=1)
            results['comet']     = float(np.mean(comet_out.scores))
            results['comet_std'] = float(np.std(comet_out.scores))
        except Exception as e:
            print(f"  Warning: COMET failed: {e}")
            results['comet'] = results['comet_std'] = None
    else:
        results['comet'] = results['comet_std'] = None

    return results


def _fmt_scores(scores):
    s = f"BLEU={scores['bleu']:.2f}  chrF={scores['chrf']:.2f}"
    if scores['comet'] is not None:
        s += f"  COMET={scores['comet']:.4f}"
    return s


# ============================================================================
# ZERO-SHOT BASELINE
# ============================================================================

def evaluate_zero_shot(test_df, model_keys, representations):
    """Evaluate each model on each representation with no fine-tuning.

    For the baseline representation this shows what pretraining alone achieves.
    For non-baseline representations (pinyin, morphemes, radicals) it shows
    how badly the model degrades without task-specific fine-tuning — establishing
    the improvement delta that fine-tuning provides.
    """
    print(f"\n{'#' * 80}")
    print("# ZERO-SHOT EVALUATION")
    print(f"{'#' * 80}\n")

    zero_shot_results = []

    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]
        print(f"  [{cfg['label']}]")
        model, tokenizer = load_model_and_tokenizer(model_key)
        model = model.cuda()

        for rep in representations:
            print(f"    {rep}... ", end='', flush=True)
            source_texts = apply_representation(test_df['chinese'].tolist(), rep)
            predictions  = translate_with_model(model, tokenizer, source_texts, model_key)
            scores       = evaluate_translations(
                predictions, test_df['english'].tolist(),
                sources=test_df['chinese'].tolist(),
            )
            print(_fmt_scores(scores))

            zero_shot_results.append({
                'condition':      'zero_shot',
                'model':          model_key,
                'representation': rep,
                'train_size':     0,
                'seed':           None,
                'bleu':           scores['bleu'],
                'bleu_std':       scores['bleu_std'],
                'chrf':           scores['chrf'],
                'comet':          scores['comet'],
                'comet_std':      scores['comet_std'],
            })

        del model
        torch.cuda.empty_cache()
        print()

    return zero_shot_results


# ============================================================================
# FINE-TUNING
# ============================================================================

def make_hf_dataset(df, representation):
    data = [
        {'translation': {
            'zh': apply_representation([row['chinese']], representation)[0],
            'en': row['english'],
        }}
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(data)


def fine_tune(train_df, val_df, representation, model_key, output_dir,
              num_epochs=5, batch_size=4, learning_rate=5e-5, seed=42):
    cfg = MODEL_CONFIGS[model_key]
    use_lora = cfg.get('use_lora', False) and _PEFT_AVAILABLE
    lora_tag = ' [LoRA]' if use_lora else ''
    print(f"\n  [{cfg['label']}]{lora_tag} {representation} | n={len(train_df)} | seed={seed}")

    model, tokenizer = load_model_and_tokenizer(model_key)

    if use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=cfg.get('lora_modules', ['q_proj', 'v_proj']),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    train_ds = make_hf_dataset(train_df, representation)
    val_ds   = make_hf_dataset(val_df,   representation)

    def preprocess(examples):
        inputs  = [ex['zh'] for ex in examples['translation']]
        targets = [ex['en'] for ex in examples['translation']]
        mi = tokenizer(inputs,  max_length=128, truncation=True, padding='max_length')
        lb = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
        mi['labels'] = lb['input_ids']
        return mi

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=['translation'])
    val_ds   = val_ds.map(preprocess,   batched=True, remove_columns=['translation'])

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to='none',
        logging_steps=10,
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(model_keys, representations, train_sizes, seeds,
                   output_dir='./models_supercomputer',
                   run_zero_shot=True, result_suffix=''):

    n_ft = len(model_keys) * len(representations) * len(train_sizes) * len(seeds)
    print(f"\n{'#' * 80}")
    print(f"# {len(model_keys)} models × {len(representations)} representations × "
          f"{len(train_sizes)} train sizes × {len(seeds)} seeds")
    print(f"# = {n_ft} fine-tuning conditions  +  zero-shot")
    print(f"{'#' * 80}\n")

    all_results = []

    # ── Zero-shot (no randomness — skipped in array tasks > 0) ───────────────
    if run_zero_shot:
        all_results.extend(evaluate_zero_shot(test_data, model_keys, representations))
        pd.DataFrame(all_results).to_csv(
            f'finetuned_results_intermediate{result_suffix}.csv', index=False
        )

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'=' * 80}")
        print(f"MODEL: {cfg['label']}")
        print(f"{'=' * 80}")

        for seed in seeds:
            for train_size in train_sizes:
                train_df = sample_train_data(remaining_data, train_size, seed=seed)

                for rep in representations:
                    out_dir = f"{output_dir}/{model_key}_{rep}_{train_size}_s{seed}"

                    model, tokenizer = fine_tune(
                        train_df, val_data,
                        representation=rep,
                        model_key=model_key,
                        output_dir=out_dir,
                        num_epochs=5,
                        batch_size=4,
                        learning_rate=5e-5,
                        seed=seed,
                    )

                    model = model.cuda()
                    source_texts = apply_representation(
                        test_data['chinese'].tolist(), rep
                    )
                    predictions = translate_with_model(
                        model, tokenizer, source_texts, model_key
                    )
                    scores = evaluate_translations(
                        predictions, test_data['english'].tolist(),
                        sources=test_data['chinese'].tolist(),
                    )
                    print(f"    → {_fmt_scores(scores)}")

                    all_results.append({
                        'condition':      'finetuned',
                        'model':          model_key,
                        'representation': rep,
                        'train_size':     train_size,
                        'seed':           seed,
                        'bleu':           scores['bleu'],
                        'bleu_std':       scores['bleu_std'],
                        'chrf':           scores['chrf'],
                        'comet':          scores['comet'],
                        'comet_std':      scores['comet_std'],
                    })

                    # Save after every condition so we don't lose progress
                    pd.DataFrame(all_results).to_csv(
                        f'finetuned_results_intermediate{result_suffix}.csv', index=False
                    )

                    # Free GPU memory + disk (checkpoints are ~300MB–2.5GB each)
                    del model, tokenizer
                    torch.cuda.empty_cache()
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)

    # ── Save & summarise ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'finetuned_results_FINAL{result_suffix}.csv', index=False)

    print(f"\n{'#' * 80}")
    print("# RESULTS SUMMARY")
    print(f"{'#' * 80}\n")

    print("Zero-shot:")
    zs = results_df[results_df['condition'] == 'zero_shot'][
        ['model', 'representation', 'bleu', 'chrf', 'comet']
    ]
    print(zs.to_string(index=False))

    print("\nFine-tuned (mean ± std across seeds):")
    ft = results_df[results_df['condition'] == 'finetuned'].copy()
    ft['comet'] = pd.to_numeric(ft['comet'], errors='coerce')
    summary = ft.groupby(['model', 'representation', 'train_size']).agg(
        bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
        chrf_mean=('chrf', 'mean'), chrf_sd=('chrf', 'std'),
        comet_mean=('comet', 'mean'), comet_sd=('comet', 'std'),
    ).reset_index()
    print(summary.to_string(index=False))

    return results_df


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    MODELS          = ['opus-mt', 'nllb-600M']
    REPRESENTATIONS = ['baseline', 'morphemes', 'pinyin', 'radicals', 'cangjie', 'wubi']
    TRAIN_SIZES     = [50, 100, 250, 500, 1000]
    ALL_SEEDS       = [42, 123, 456]
    OUTPUT_DIR      = './models_supercomputer'

    # ── Job-array mode: $SLURM_ARRAY_TASK_ID selects the seed ────────────────
    # Each array task (0, 1, 2) runs one seed; task 0 also handles zero-shot.
    # ── Standalone mode: no env var → run all seeds in series ────────────────
    array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))

    if array_task_id >= 0:
        seeds         = [ALL_SEEDS[array_task_id]]
        run_zero_shot = (array_task_id == 0)
        suffix        = f'_s{seeds[0]}'
        print(f'Job-array mode: task {array_task_id}, seed {seeds[0]}, '
              f'zero_shot={run_zero_shot}\n')
    else:
        seeds         = ALL_SEEDS
        run_zero_shot = True
        suffix        = ''
        print('Standalone mode: running all seeds\n')

    results = run_experiment(
        model_keys=MODELS,
        representations=REPRESENTATIONS,
        train_sizes=TRAIN_SIZES,
        seeds=seeds,
        output_dir=OUTPUT_DIR,
        run_zero_shot=run_zero_shot,
        result_suffix=suffix,
    )

    print(f'\n✓ Results saved to: finetuned_results_FINAL{suffix}.csv')
    print('\nDone!')
