"""
run_pooled_bootstrap.py — Corpus-level bootstrap by pooling predictions across seeds.

Per-seed bootstrap doesn't work well (n=500, sent_bleu ≠ corpus BLEU). This script
pools all 5 seeds' predictions per condition into a single set (n=2500), then
bootstraps over those for a proper corpus-level significance estimate.

Compares each rep vs. baseline; reports both BLEU and COMET deltas.

Usage:
    PYTHONPATH=src python src/run_pooled_bootstrap.py \
        --preds_dir results/predictions_v3/ \
        --out results/pooled_bootstrap.csv
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))


def parse_pred_filename(stem):
    # handles both regular and unseen: strips _unseen_preds or _preds
    stem = stem.replace('_unseen_preds', '').replace('_preds', '')
    is_unseen = '_unseen_preds' in Path(stem).name or 'unseen' in stem
    parts = stem.split('_')
    seed_part = [p for p in parts if p.startswith('s') and p[1:].isdigit()]
    if not seed_part:
        return None
    idx = parts.index(seed_part[0])
    return {
        'model':      '_'.join(parts[:idx-2]),
        'rep':        parts[idx-2],
        'train_size': int(parts[idx-1]),
        'seed':       int(seed_part[0][1:]),
    }


def bootstrap_corpus_bleu(base_sents, rep_sents, n_iter=1000, rng=None):
    """
    Bootstrap corpus BLEU by resampling sentence pairs.
    Uses sacrebleu sentence_bleu as a proxy, then takes corpus mean.
    Returns delta, p-value (one-sided), 95% CI.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    base_arr = np.array(base_sents)
    rep_arr  = np.array(rep_sents)
    obs_delta = rep_arr.mean() - base_arr.mean()
    diffs = rep_arr - base_arr
    samples = rng.choice(diffs, size=(n_iter, len(diffs)), replace=True)
    boot_deltas = samples.mean(axis=1)
    p = float((boot_deltas > obs_delta).sum()) / n_iter
    ci = (float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5)))
    return {
        'base': float(base_arr.mean()), 'rep': float(rep_arr.mean()),
        'delta': float(obs_delta), 'p': p,
        'ci_lo': ci[0], 'ci_hi': ci[1],
        'significant': p < 0.05,
        'n': len(diffs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir', default='results/predictions_v3/')
    parser.add_argument('--out', default='results/pooled_bootstrap.csv')
    parser.add_argument('--n_iter', type=int, default=1000)
    args = parser.parse_args()

    # Load all prediction files, split regular vs unseen
    groups = defaultdict(lambda: defaultdict(list))  # (test_set, model, train_size, rep) → [data]

    for fpath in sorted(glob.glob(os.path.join(args.preds_dir, '*.json'))):
        fname = Path(fpath).stem
        is_unseen = 'unseen' in fname
        test_set = 'unseen' if is_unseen else 'regular'
        stem = fname.replace('_unseen_preds', '').replace('_preds', '')
        meta = parse_pred_filename(stem)
        if meta is None:
            continue
        with open(fpath) as f:
            data = json.load(f)
        key = (test_set, meta['model'], meta['train_size'], meta['rep'])
        groups[key]['sent_bleus'].extend(data.get('sent_bleus', []))
        groups[key]['comet_scores'].extend([x for x in (data.get('comet_scores') or [])
                                             if x is not None])

    print(f"Loaded {len(groups)} pooled conditions")

    rng = np.random.RandomState(42)
    records = []

    for test_set in ['regular', 'unseen']:
        base_keys = [(ts, m, sz, 'baseline') for ts, m, sz, r in groups
                     if ts == test_set and r == 'baseline']

        print(f"\n{'='*100}")
        print(f"TEST SET: {test_set.upper()}  (pooled n per condition)")
        print(f"{'='*100}")

        for bkey in sorted(base_keys):
            _, model, train_size, _ = bkey
            if bkey not in groups:
                continue
            base = groups[bkey]

            for rep in ['morphemes', 'sentencepiece', 'radicals']:
                rkey = (test_set, model, train_size, rep)
                if rkey not in groups:
                    continue
                rd = groups[rkey]

                row = {'test_set': test_set, 'model': model,
                       'train_size': train_size, 'rep': rep}

                for metric, bscores, rscores in [
                    ('bleu',  base['sent_bleus'],    rd['sent_bleus']),
                    ('comet', base['comet_scores'],   rd['comet_scores']),
                ]:
                    if not bscores or not rscores:
                        row[f'{metric}_delta'] = row[f'{metric}_p'] = None
                        row[f'{metric}_sig'] = row[f'{metric}_ci_lo'] = row[f'{metric}_ci_hi'] = None
                        continue
                    n = min(len(bscores), len(rscores))
                    r = bootstrap_corpus_bleu(bscores[:n], rscores[:n],
                                              n_iter=args.n_iter, rng=rng)
                    row[f'{metric}_base']  = round(r['base'],  4)
                    row[f'{metric}_rep']   = round(r['rep'],   4)
                    row[f'{metric}_delta'] = round(r['delta'], 4)
                    row[f'{metric}_p']     = round(r['p'],     4)
                    row[f'{metric}_sig']   = r['significant']
                    row[f'{metric}_ci_lo'] = round(r['ci_lo'], 4)
                    row[f'{metric}_ci_hi'] = round(r['ci_hi'], 4)
                    row[f'{metric}_n']     = r['n']

                sig_b = '***' if row.get('bleu_sig')  else '   '
                sig_c = '***' if row.get('comet_sig') else '   '
                b_str = (f"ΔBLEU={row['bleu_delta']:+.4f}{sig_b}  p={row['bleu_p']:.3f}"
                         f"  CI=[{row['bleu_ci_lo']:+.4f},{row['bleu_ci_hi']:+.4f}]"
                         if row.get('bleu_delta') is not None else "BLEU=N/A")
                c_str = (f"  ΔCOMET={row['comet_delta']:+.4f}{sig_c}  p={row['comet_p']:.3f}"
                         if row.get('comet_delta') is not None else "")
                print(f"  {model:9s} | {rep:14s} | n={train_size:4d} |"
                      f" n_sents={row.get('bleu_n','?'):5} | {b_str}{c_str}")
                records.append(row)

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n✓ Saved pooled bootstrap to {args.out}")

    # Highlight significant results
    sig = df[df['bleu_sig'] == True]
    if not sig.empty:
        print("\nSIGNIFICANT BLEU results (p<0.05):")
        print(sig[['test_set','model','rep','train_size','bleu_delta','bleu_p','comet_delta','comet_p']].to_string())
    else:
        print("\nNo significant BLEU results at p<0.05 (expected — metric inflation is corpus-level).")

    sig_c = df[df['comet_sig'] == True]
    if not sig_c.empty:
        print("\nSIGNIFICANT COMET results (p<0.05):")
        print(sig_c[['test_set','model','rep','train_size','comet_delta','comet_p']].to_string())


if __name__ == '__main__':
    main()
