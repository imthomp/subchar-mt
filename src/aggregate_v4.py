"""
aggregate_v4.py — Aggregate v4 selective-decomposition results.

Combines per-seed CSVs, prints summary, and highlights the key comparison:
  selective_radicals vs. radicals vs. baseline
on both regular and unseen-character test sets.

Usage:
    python src/aggregate_v4.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    files = sorted(glob.glob(str(PROJECT_DIR / 'v4_results_FINAL_s*.csv')))
    if not files:
        print("No v4_results_FINAL_s*.csv files found. Jobs may not be done yet.")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['comet'] = pd.to_numeric(df['comet'], errors='coerce')
    print(f"Loaded {len(files)} seed files → {len(df)} rows")

    out = PROJECT_DIR / 'v4_results_ALL.csv'
    df.to_csv(out, index=False)

    for test_set in ['regular', 'unseen']:
        sub = df[df['test_set'] == test_set]
        if sub.empty:
            continue
        summ = sub.groupby(['model', 'representation', 'train_size']).agg(
            bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
            chrf_mean=('chrf', 'mean'),
            comet_mean=('comet', 'mean'), comet_sd=('comet', 'std'),
            n=('bleu', 'count'),
        ).round(3)

        print(f"\n{'='*80}")
        print(f"TEST SET: {test_set.upper()}")
        print(f"{'='*80}")
        print(summ.to_string())

    # Key comparison: selective vs full decomposition
    print(f"\n{'='*80}")
    print("KEY COMPARISON: selective vs full decomposition (regular test set, mean over train sizes)")
    print(f"{'='*80}")
    reg = df[df['test_set'] == 'regular']
    key_reps = ['baseline', 'radicals', 'selective_radicals', 'morphemes', 'selective_morphemes']
    key = reg[reg['representation'].isin(key_reps)].groupby(
        ['model', 'representation']
    ).agg(
        bleu_mean=('bleu', 'mean'),
        comet_mean=('comet', 'mean'),
    ).round(3)
    print(key.to_string())

    print(f"\n✓ Combined results saved to {out}")


if __name__ == '__main__':
    main()
