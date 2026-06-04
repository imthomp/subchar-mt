"""
aggregate_v5.py — Aggregate v5 decomposition-depth ablation results.

Key question: does BLEU/COMET vary by IDS decomposition depth?
Han et al. (arXiv 2512.15556) found rxd2 collapses; we test the same here.

Usage:
    python src/aggregate_v5.py
"""
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEPTH_ORDER = ['baseline', 'radicals_d1', 'radicals_d2', 'radicals_full']
DEPTH_LABELS = {
    'baseline':      'Baseline (chars)',
    'radicals_d1':   'Radicals depth-1 (ideograph)',
    'radicals_d2':   'Radicals depth-2 (intermediate)',
    'radicals_full': 'Radicals full (near-primitive)',
}

def main():
    files = sorted(glob.glob(str(PROJECT_DIR / 'v5_results_FINAL_s*.csv')))
    if not files:
        print("No v5_results_FINAL_s*.csv found — job may not be done yet.")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['comet'] = pd.to_numeric(df['comet'], errors='coerce')
    print(f"Loaded {len(files)} seed files → {len(df)} rows\n")
    df.to_csv(PROJECT_DIR / 'v5_results_ALL.csv', index=False)

    summ = df.groupby(['model', 'representation', 'train_size']).agg(
        bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
        chrf_mean=('chrf', 'mean'),
        comet_mean=('comet', 'mean'), comet_sd=('comet', 'std'),
        n=('bleu', 'count'),
    ).round(3)

    print('=' * 80)
    print('DECOMPOSITION DEPTH ABLATION — full results')
    print('=' * 80)
    print(summ.to_string())

    # Depth comparison: mean over all train sizes
    print('\n' + '=' * 80)
    print('DEPTH EFFECT (mean over train sizes)')
    print('=' * 80)
    depth_summ = df.groupby(['model', 'representation']).agg(
        bleu=('bleu', 'mean'), chrf=('chrf', 'mean'), comet=('comet', 'mean')
    ).round(3)

    for model in ['opus-mt', 'nllb-600M']:
        print(f'\n  {model}')
        print(f"  {'Representation':35s}  {'BLEU':>7}  {'chrF':>7}  {'COMET':>7}")
        print('  ' + '-' * 58)
        for rep in DEPTH_ORDER:
            try:
                row = depth_summ.loc[(model, rep)]
                label = DEPTH_LABELS.get(rep, rep)
                print(f"  {label:35s}  {row['bleu']:>7.2f}  {row['chrf']:>7.2f}  {row['comet']:>7.3f}")
            except KeyError:
                pass

    # Delta vs baseline
    print('\n' + '=' * 80)
    print('DELTA vs. BASELINE')
    print('=' * 80)
    for model in ['opus-mt', 'nllb-600M']:
        print(f'\n  {model}')
        try:
            base = depth_summ.loc[(model, 'baseline')]
        except KeyError:
            continue
        for rep in ['radicals_d1', 'radicals_d2', 'radicals_full']:
            try:
                row = depth_summ.loc[(model, rep)]
                label = DEPTH_LABELS.get(rep, rep)
                print(f"  {label:35s}  ΔBLEU={row['bleu']-base['bleu']:+.2f}  "
                      f"ΔchrF={row['chrf']-base['chrf']:+.2f}  "
                      f"ΔCOMET={row['comet']-base['comet']:+.3f}")
            except KeyError:
                pass

    print(f"\n✓ Saved v5_results_ALL.csv")

if __name__ == '__main__':
    main()
