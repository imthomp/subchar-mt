"""
aggregate_zhja.py — Aggregate Zh→Ja CJK extension results.

Key question: does radical decomposition help Zh→Ja MORE than Zh→En?
Compares radicals vs. baseline delta across both language pairs.

Usage:
    python src/aggregate_zhja.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    files = sorted(glob.glob(str(PROJECT_DIR / 'zhja_results_FINAL_s*.csv')))
    if not files:
        print("No zhja_results_FINAL_s*.csv files found.")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['comet'] = pd.to_numeric(df['comet'], errors='coerce')
    print(f"Loaded {len(files)} seed files → {len(df)} rows\n")
    df.to_csv(PROJECT_DIR / 'zhja_results_ALL.csv', index=False)

    summ = df.groupby(['model', 'representation', 'train_size']).agg(
        bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
        chrf_mean=('chrf', 'mean'),
        comet_mean=('comet', 'mean'), comet_sd=('comet', 'std'),
        n=('bleu', 'count'),
    ).round(3)

    print("=" * 80)
    print("Zh→Ja RESULTS (FLORES+ test set, 1012 sentences)")
    print("=" * 80)
    print(summ.to_string())

    # Delta vs baseline
    print("\n" + "=" * 80)
    print("DELTA vs. BASELINE per model × train_size")
    print("=" * 80)
    baseline = summ.xs('baseline', level='representation')[['bleu_mean', 'comet_mean']]
    for rep in ['morphemes', 'radicals', 'sentencepiece']:
        try:
            rep_df = summ.xs(rep, level='representation')[['bleu_mean', 'comet_mean']]
            delta = (rep_df - baseline).round(3)
            delta.columns = [f'{rep}_Δbleu', f'{rep}_Δcomet']
            print(f"\n{rep}:")
            print(delta.to_string())
        except KeyError:
            pass

    # Cross-language comparison: load v3 regular for reference
    v3_files = sorted(glob.glob(str(PROJECT_DIR / 'v3_results_FINAL_s*.csv')))
    if v3_files:
        v3 = pd.concat([pd.read_csv(f) for f in v3_files], ignore_index=True)
        v3 = v3[v3['test_set'] == 'regular']
        v3['comet'] = pd.to_numeric(v3['comet'], errors='coerce')
        v3_summ = v3.groupby(['model', 'representation']).agg(
            bleu=('bleu', 'mean'), comet=('comet', 'mean')
        ).round(3)
        zhen_base = v3_summ.xs('baseline', level='representation')
        zhen_rad  = v3_summ.xs('radicals', level='representation')

        zhja_summ2 = df.groupby(['model', 'representation']).agg(
            bleu=('bleu', 'mean'), comet=('comet', 'mean')
        ).round(3)
        zhja_base = zhja_summ2.xs('baseline', level='representation')
        zhja_rad  = zhja_summ2.xs('radicals', level='representation')

        print("\n" + "=" * 80)
        print("CROSS-LANGUAGE: radical decomposition delta (radicals − baseline)")
        print("=" * 80)
        print(f"{'model':15s} {'Zh→En ΔBLEU':>12} {'Zh→En ΔCOMET':>13} "
              f"{'Zh→Ja ΔBLEU':>12} {'Zh→Ja ΔCOMET':>13}")
        for model in zhen_base.index:
            if model not in zhja_base.index:
                continue
            d_en_bleu  = round(zhen_rad.loc[model, 'bleu']  - zhen_base.loc[model, 'bleu'],  3)
            d_en_comet = round(zhen_rad.loc[model, 'comet'] - zhen_base.loc[model, 'comet'], 4)
            d_ja_bleu  = round(zhja_rad.loc[model, 'bleu']  - zhja_base.loc[model, 'bleu'],  3)
            d_ja_comet = round(zhja_rad.loc[model, 'comet'] - zhja_base.loc[model, 'comet'], 4)
            print(f"{model:15s} {d_en_bleu:>12} {d_en_comet:>13} "
                  f"{d_ja_bleu:>12} {d_ja_comet:>13}")

    print(f"\n✓ Saved zhja_results_ALL.csv")


if __name__ == '__main__':
    main()
