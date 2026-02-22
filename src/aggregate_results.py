"""
Aggregate per-seed CSVs produced by the job array into a single summary.

Usage (after all array tasks finish):
    python aggregate_results.py

Outputs:
    finetuned_results_ALL.csv       — all rows from all seeds combined
    finetuned_results_summary.csv   — mean ± sd across seeds per condition
"""

import glob
import pandas as pd
import numpy as np

# ── Load all per-seed files ───────────────────────────────────────────────────
seed_files = sorted(glob.glob('finetuned_results_FINAL_s*.csv'))
if not seed_files:
    raise FileNotFoundError(
        "No per-seed result files found (finetuned_results_FINAL_s*.csv). "
        "Have all array tasks finished?"
    )

print(f"Found {len(seed_files)} seed file(s): {seed_files}")
dfs = [pd.read_csv(f) for f in seed_files]
combined = pd.concat(dfs, ignore_index=True)

# Remove duplicate zero-shot rows (only task 0 writes them, but guard anyway)
zs_mask = combined['condition'] == 'zero_shot'
combined = pd.concat([
    combined[zs_mask].drop_duplicates(subset=['model', 'representation']),
    combined[~zs_mask],
], ignore_index=True)

combined.to_csv('finetuned_results_ALL.csv', index=False)
print(f"✓ Combined: {len(combined)} rows → finetuned_results_ALL.csv")

# ── Summary table ─────────────────────────────────────────────────────────────
ft = combined[combined['condition'] == 'finetuned'].copy()
ft['comet'] = pd.to_numeric(ft['comet'], errors='coerce')

summary = ft.groupby(['model', 'representation', 'train_size']).agg(
    n_seeds=('seed', 'count'),
    bleu_mean=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
    chrf_mean=('chrf', 'mean'), chrf_sd=('chrf', 'std'),
    comet_mean=('comet', 'mean'), comet_sd=('comet', 'std'),
).reset_index()

# Round for readability
for col in ['bleu_mean', 'bleu_sd', 'chrf_mean', 'chrf_sd']:
    summary[col] = summary[col].round(2)
for col in ['comet_mean', 'comet_sd']:
    summary[col] = summary[col].round(4)

summary.to_csv('finetuned_results_summary.csv', index=False)
print(f"✓ Summary:  {len(summary)} conditions → finetuned_results_summary.csv\n")

# ── Print tables ──────────────────────────────────────────────────────────────
print("=" * 80)
print("ZERO-SHOT RESULTS")
print("=" * 80)
zs = combined[combined['condition'] == 'zero_shot'][
    ['model', 'representation', 'bleu', 'chrf', 'comet']
].round(3)
print(zs.to_string(index=False))

print()
print("=" * 80)
print("FINE-TUNED RESULTS (mean ± sd across seeds)")
print("=" * 80)
print(summary.to_string(index=False))

# ── Best per metric ───────────────────────────────────────────────────────────
print()
print("=" * 80)
print("BEST CONDITIONS PER METRIC")
print("=" * 80)
for metric in ['bleu_mean', 'chrf_mean', 'comet_mean']:
    best = summary.loc[summary[metric].idxmax()]
    print(f"  {metric:12s}: {best['model']} / {best['representation']} "
          f"@ {best['train_size']} examples  "
          f"({metric}={best[metric]:.3f})")
