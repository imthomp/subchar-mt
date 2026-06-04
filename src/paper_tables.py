"""
paper_tables.py — Generate draft-quality results tables for the paper.

Pulls from all completed experiments and prints LaTeX-ready tables + plain summaries.

Tables produced:
  1. Main results (v3): BLEU / chrF / COMET by rep × model × train_size
  2. Unseen-character results (v3): same, unseen test set
  3. Transparency stratification (HKCCPN): BLEU/chrF by stratum × rep
  4. Probing results: frequency-bin probe accuracy by rep
  5. Cross-language table (v3 Zh→En vs Zh→Ja, once zhja done)
  6. Selective decomp (v4, once done)

Usage:
    python src/paper_tables.py [--latex]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent


REP_ORDER = ['baseline', 'morphemes', 'sentencepiece', 'radicals',
             'selective_radicals', 'selective_morphemes']
REP_LABELS = {
    'baseline':           'Baseline (chars)',
    'morphemes':          'Morphemes (jieba)',
    'sentencepiece':      'SentencePiece',
    'radicals':           'Radicals (IDS)',
    'selective_radicals': 'Selective radicals',
    'selective_morphemes':'Selective morphemes',
    'random_index':       'Random index',
    'byte':               'Byte encoding',
}
MODEL_LABELS = {
    'opus-mt':   'opus-mt',
    'nllb-600M': 'NLLB-600M',
}


def load_v3():
    files = sorted(glob.glob(str(PROJECT_DIR / 'v3_results_FINAL_s*.csv')))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['comet'] = pd.to_numeric(df['comet'], errors='coerce')
    return df


def load_v4():
    files = sorted(glob.glob(str(PROJECT_DIR / 'v4_results_FINAL_s*.csv')))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['comet'] = pd.to_numeric(df['comet'], errors='coerce')
    return df


def load_zhja():
    files = sorted(glob.glob(str(PROJECT_DIR / 'zhja_results_FINAL_s*.csv')))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['comet'] = pd.to_numeric(df['comet'], errors='coerce')
    return df


def load_probing():
    p = PROJECT_DIR / 'results' / 'probing' / 'probe_results.csv'
    return pd.read_csv(p) if p.exists() else None


def load_transparency():
    p = PROJECT_DIR / 'results' / 'stratified_analysis_hkccpn.csv'
    return pd.read_csv(p) if p.exists() else None


def fmt(val, sd=None, bold_thresh=None, is_comet=False):
    if pd.isna(val):
        return '—'
    s = f'{val:.2f}' if not is_comet else f'{val:.3f}'
    if sd is not None and not pd.isna(sd):
        s += f'$_{{\\pm{sd:.2f}}}$'
    return s


def table_main(df, test_set='regular', latex=False):
    sub = df[df['test_set'] == test_set]
    summ = sub.groupby(['model', 'representation', 'train_size']).agg(
        bleu=('bleu', 'mean'), bleu_sd=('bleu', 'std'),
        chrf=('chrf', 'mean'),
        comet=('comet', 'mean'), comet_sd=('comet', 'std'),
    ).round(3).reset_index()

    title = f"Table: {test_set.upper()} test set results (mean over 5 seeds)"
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")

    for model in ['opus-mt', 'nllb-600M']:
        print(f"\n  Model: {MODEL_LABELS.get(model, model)}")
        m = summ[summ['model'] == model]

        header = f"  {'Rep':22s} {'n':>5}  {'BLEU':>7}  {'chrF':>7}  {'COMET':>7}"
        print(header)
        print('  ' + '-' * (len(header) - 2))

        reps = [r for r in REP_ORDER if r in m['representation'].values]
        for rep in reps:
            for n in sorted(m['train_size'].unique()):
                row = m[(m['representation'] == rep) & (m['train_size'] == n)]
                if row.empty:
                    continue
                r = row.iloc[0]
                label = REP_LABELS.get(rep, rep) if n == m['train_size'].min() else ''
                print(f"  {label:22s} {int(n):>5}  "
                      f"{r['bleu']:>7.2f}  {r['chrf']:>7.2f}  "
                      f"{r['comet']:>7.3f}" if not pd.isna(r['comet']) else
                      f"  {label:22s} {int(n):>5}  {r['bleu']:>7.2f}  {r['chrf']:>7.2f}  {'—':>7}")


def table_probing(df_probe):
    print(f"\n{'='*70}")
    print("Table: Frequency-bin probing accuracy (Δ over majority baseline = 0.333)")
    print(f"{'='*70}")
    summ = df_probe[df_probe['task'] == 'frequency'].groupby(['model', 'rep']).agg(
        delta=('delta_acc', 'mean'),
        acc=('accuracy', 'mean'),
    ).round(4).reset_index()

    for model in ['opus-mt', 'nllb-600M']:
        print(f"\n  {MODEL_LABELS.get(model, model)}")
        m = summ[summ['model'] == model].sort_values('delta', ascending=False)
        print(f"  {'Rep':22s}  {'Δ acc':>8}  {'acc':>8}")
        print('  ' + '-' * 42)
        for _, row in m.iterrows():
            label = REP_LABELS.get(row['rep'], row['rep'])
            marker = ' ← lowest' if row['delta'] == m['delta'].min() else ''
            print(f"  {label:22s}  {row['delta']:>8.4f}  {row['acc']:>8.4f}{marker}")


def table_transparency(df_trans):
    print(f"\n{'='*70}")
    print("Table: HKCCPN transparency stratification (mean BLEU / chrF)")
    print(f"{'='*70}")
    strata_order = ['low', 'mid', 'high', 'unseen', 'seen']
    summ = df_trans.groupby(['model', 'representation', 'stratum']).agg(
        bleu=('bleu_mean', 'mean'),
        chrf=('chrf_mean', 'mean'),
    ).round(2).reset_index()

    for model in ['opus-mt', 'nllb-600M']:
        print(f"\n  {MODEL_LABELS.get(model, model)}")
        m = summ[summ['model'] == model]
        reps = [r for r in REP_ORDER if r in m['representation'].values]
        strata = [s for s in strata_order if s in m['stratum'].values]

        header = f"  {'Rep':22s}" + ''.join(f"  {s:>8}" for s in strata)
        print(header + '  (BLEU)')
        print('  ' + '-' * (len(header) - 2 + 10 * len(strata)))
        for rep in reps:
            label = REP_LABELS.get(rep, rep)
            row_str = f"  {label:22s}"
            for s in strata:
                cell = m[(m['representation'] == rep) & (m['stratum'] == s)]
                row_str += f"  {cell['bleu'].values[0]:>8.2f}" if not cell.empty else '  ' + '—'.rjust(8)
            print(row_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latex', action='store_true')
    args = parser.parse_args()

    v3      = load_v3()
    v4      = load_v4()
    zhja    = load_zhja()
    probe   = load_probing()
    trans   = load_transparency()

    print("\n" + "=" * 90)
    print("PAPER TABLES — subchar-mt")
    print("=" * 90)

    if v3 is not None:
        table_main(v3, test_set='regular', latex=args.latex)
        table_main(v3, test_set='unseen',  latex=args.latex)
    else:
        print("\n[v3 results not found]")

    if probe is not None:
        table_probing(probe)
    else:
        print("\n[probing results not found]")

    if trans is not None:
        table_transparency(trans)
    else:
        print("\n[transparency results not found]")

    if v4 is not None:
        print(f"\n{'='*70}")
        print("Table: Selective decomposition (v4)")
        print(f"{'='*70}")
        table_main(v4, test_set='regular', latex=args.latex)
        table_main(v4, test_set='unseen',  latex=args.latex)
    else:
        print("\n[v4 results not ready yet]")

    if zhja is not None:
        print(f"\n{'='*70}")
        print("Table: Zh→Ja CJK extension")
        print(f"{'='*70}")
        summ = zhja.groupby(['model', 'representation', 'train_size']).agg(
            bleu=('bleu', 'mean'), chrf=('chrf', 'mean'), comet=('comet', 'mean')
        ).round(3)
        print(summ.to_string())
    else:
        print("\n[Zh→Ja results not ready yet]")


if __name__ == '__main__':
    main()
