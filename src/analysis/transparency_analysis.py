"""
transparency_analysis.py — Stratified evaluation by semantic-radical transparency.

This is the "key novel result" the roadmap identifies: radical decomposition should help
semantically transparent characters (where the semantic radical predicts meaning) and hurt
opaque ones (where it doesn't). A significant transparency × representation interaction
is the finding that elevates the paper from a horse-race to an explanation.

Two analysis modes:

  1. HKCCPN-based stratification (preferred)
     Uses the Hong Kong Chinese Character Psycholinguistic Norms (Su, Yum & Lau 2022,
     Behavior Research Methods, DOI 10.3758/s13428-022-01928-y) which provides subjective
     transparency ratings for 4,376 characters. Download the norms and point --hkccpn_csv
     at the file.

  2. Frequency-based stratification (fallback, available now)
     Stratifies test-set characters by token frequency in the training corpus. High-frequency
     characters are "seen" characters; low-frequency/absent characters are the unseen-character
     condition from Saunders et al. (WAT 2020).

Usage:
    # Frequency-based (no external data needed):
    python src/analysis/transparency_analysis.py \\
        --preds_dir results/predictions/ \\
        --train_csv finetuned_results_ALL.csv

    # With HKCCPN norms:
    python src/analysis/transparency_analysis.py \\
        --preds_dir results/predictions/ \\
        --hkccpn_csv data/hkccpn_norms.csv

    # Build unseen-character test set:
    python src/analysis/transparency_analysis.py \\
        --build_unseen_test \\
        --out data/unseen_char_test.csv
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))


# ── Radical / component lookup ────────────────────────────────────────────────

def load_kangxi_radical_map(ids_path: Optional[str] = None) -> Dict[str, str]:
    """
    Build a char → semantic_radical mapping.
    For phono-semantic compounds (~80% of Chinese characters), the semantic radical
    (偏旁部首) is typically the left/top component in a ⿰/⿱ decomposition.
    This is a heuristic — proper radical assignment requires the Kangxi table.
    """
    # Kangxi radical Unicode block: U+2F00–U+2FD5 (214 radicals)
    kangxi_start = 0x2F00

    # Try to load the Unihan kRSKangXi field from ids.txt heuristically
    radical_map: Dict[str, str] = {}
    ids_map: Dict[str, str] = {}

    ids_path = ids_path or str(PROJECT_DIR / "data" / "ids.txt")
    if os.path.exists(ids_path):
        with open(ids_path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                char = parts[1]
                decomp = parts[2]
                # Strip source tags [GTV] etc.
                clean, skip = '', False
                for ch in decomp:
                    if ch == '[': skip = True
                    elif ch == ']': skip = False
                    elif not skip: clean += ch
                ids_map[char] = clean.strip()

        # Heuristic: for ⿰ (left-right) compounds, left component is often semantic radical
        layout_markers = set('⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻')
        for char, decomp in ids_map.items():
            comps = [c for c in decomp if c not in layout_markers]
            if len(comps) >= 1:
                radical_map[char] = comps[0]

    return radical_map, ids_map


# ── HKCCPN norms ─────────────────────────────────────────────────────────────

def load_hkccpn(path: str) -> pd.DataFrame:
    """
    Load the Hong Kong Chinese Character Psycholinguistic Norms.
    Su, Yum & Lau (2022), Behavior Research Methods.
    DOI: 10.3758/s13428-022-01928-y
    Download: https://mst-cbs.polyu.edu.hk/Database/HK_RatingsNorm_2022.xlsx

    Accepts .xlsx (original) or .csv. Characters are Traditional Chinese;
    we add a Simplified Chinese index via opencc for matching WMT19 sources.

    Returns a DataFrame indexed by character with columns:
      'transparency' (SemanticRadicalTrans, normalized 0–1),
      'familiarity', 'freq_log', 'aoa'
    Both traditional and simplified forms are indexed.
    """
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding='utf-8-sig')

    # Known column mapping for HK_RatingsNorm_2022.xlsx
    col_map = {
        'Character': 'character',
        'SemanticRadicalTrans': 'transparency',
        'Familarity': 'familiarity',  # note: typo in source file
        'Frequency.Log': 'freq_log',
        'AoA': 'aoa',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Fallback flexible matching if column names differ
    if 'character' not in df.columns:
        char_col = next((c for c in df.columns if 'char' in c.lower()), None)
        if char_col:
            df = df.rename(columns={char_col: 'character'})
    if 'transparency' not in df.columns:
        trans_col = next((c for c in df.columns
                          if 'trans' in c.lower() or 'semantic' in c.lower()), None)
        if trans_col:
            df = df.rename(columns={trans_col: 'transparency'})

    if 'character' not in df.columns or 'transparency' not in df.columns:
        raise ValueError(
            f"Cannot find character/transparency columns.\nAvailable: {list(df.columns)}"
        )

    df['transparency'] = pd.to_numeric(df['transparency'], errors='coerce')
    df = df.dropna(subset=['character', 'transparency']).copy()

    # Normalize transparency to 0–1
    lo, hi = df['transparency'].min(), df['transparency'].max()
    if hi > lo:
        df['transparency'] = (df['transparency'] - lo) / (hi - lo)

    # Add Simplified Chinese equivalents via opencc (HKCCPN uses Traditional)
    try:
        import opencc
        converter = opencc.OpenCC('t2s')
        df['char_simplified'] = df['character'].apply(
            lambda c: converter.convert(c) if isinstance(c, str) else c
        )
        # Build a lookup that covers both trad and simp forms
        keep_cols = ['transparency'] + [c for c in ['familiarity', 'freq_log', 'aoa']
                                          if c in df.columns]
        trad_index = df.set_index('character')[keep_cols]
        simp_index = df.set_index('char_simplified')[keep_cols]
        combined = pd.concat([trad_index, simp_index[
            ~simp_index.index.isin(trad_index.index)
        ]])
        combined = combined[~combined.index.duplicated(keep='first')]
        print(f"  HKCCPN: {len(trad_index)} trad + {len(combined)-len(trad_index)} "
              f"simp-only = {len(combined)} total character entries")
        return combined
    except ImportError:
        print("  opencc not installed — using Traditional Chinese index only")
        return df.set_index('character')[['transparency']]


# ── Frequency stratification ──────────────────────────────────────────────────

def build_char_frequency_map(train_csv: str) -> Dict[str, int]:
    """Build a character frequency map from WMT19 (via the results CSV source column)."""
    try:
        from datasets import load_dataset
        dataset = load_dataset('wmt19', 'zh-en', split='train')
        counter: collections.Counter = collections.Counter()
        for ex in dataset:
            for ch in ex['translation']['zh']:
                counter[ch] += 1
        return dict(counter)
    except Exception as e:
        print(f"  Warning: could not load WMT19 for frequency map: {e}")
        return {}


# ── Per-sentence analysis ────────────────────────────────────────────────────

def stratify_sentences_by_transparency(
    sources: List[str],
    hkccpn: Optional[pd.DataFrame] = None,
    freq_map: Optional[Dict[str, int]] = None,
    n_quantiles: int = 3,
) -> List[str]:
    """
    Assign each source sentence a transparency stratum label.

    If HKCCPN is available: average per-character transparency ratings.
    Otherwise: average per-character log-frequency (proxy for familiarity/transparency).

    Returns a list of stratum labels: 'low', 'mid', 'high'.
    """
    scores = []
    for sent in sources:
        chars = [c for c in sent if '一' <= c <= '鿿']  # CJK Unified
        if not chars:
            scores.append(float('nan'))
            continue

        if hkccpn is not None:
            vals = [hkccpn.loc[c, 'transparency'] for c in chars if c in hkccpn.index]
            score = np.mean(vals) if vals else float('nan')
        elif freq_map:
            vals = [np.log1p(freq_map.get(c, 0)) for c in chars]
            score = np.mean(vals) if vals else float('nan')
        else:
            score = float('nan')
        scores.append(score)

    scores_arr = np.array(scores)
    finite = scores_arr[np.isfinite(scores_arr)]
    if len(finite) == 0:
        return ['unknown'] * len(sources)

    thresholds = np.percentile(finite, [100 / n_quantiles * i for i in range(1, n_quantiles)])
    labels = []
    for s in scores_arr:
        if not np.isfinite(s):
            labels.append('unknown')
        elif s <= thresholds[0]:
            labels.append('low')
        elif n_quantiles > 2 and s <= thresholds[1]:
            labels.append('mid')
        else:
            labels.append('high')
    return labels


def unseen_character_mask(sources: List[str], freq_map: Dict[str, int],
                          freq_threshold: int = 5) -> List[bool]:
    """
    Returns True for sentences containing at least one character
    that appeared fewer than freq_threshold times in the training corpus.
    These are the "unseen/rare character" sentences most likely to benefit from
    sub-character decomposition (Saunders et al. WAT 2020).
    """
    mask = []
    for sent in sources:
        cjk = [c for c in sent if '一' <= c <= '鿿']
        has_rare = any(freq_map.get(c, 0) < freq_threshold for c in cjk)
        mask.append(has_rare)
    return mask


# ── Per-stratum metric computation ───────────────────────────────────────────

def compute_stratified_metrics(
    predictions: List[str],
    references: List[str],
    strata: List[str],
    comet_scores: Optional[List[float]] = None,
) -> pd.DataFrame:
    from sacrebleu import corpus_bleu
    from sacrebleu.metrics import CHRF

    unique_strata = sorted(set(s for s in strata if s != 'unknown'))
    records = []

    for stratum in unique_strata:
        idx = [i for i, s in enumerate(strata) if s == stratum]
        if not idx:
            continue
        preds_s = [predictions[i] for i in idx]
        refs_s  = [references[i]  for i in idx]

        bleu = corpus_bleu(preds_s, [[r] for r in refs_s]).score
        chrf = CHRF().corpus_score(preds_s, [[r] for r in refs_s]).score
        comet = float(np.mean([comet_scores[i] for i in idx])) if comet_scores else None

        records.append({
            'stratum': stratum,
            'n': len(idx),
            'bleu': round(bleu, 2),
            'chrf': round(chrf, 2),
            'comet': round(comet, 4) if comet is not None else None,
        })

    return pd.DataFrame(records)


# ── Build unseen-character test set ──────────────────────────────────────────

def build_unseen_test_set(
    out_csv: str,
    min_rare_chars: int = 2,
    rare_threshold: int = 10,
    n_samples: int = 200,
) -> None:
    """
    Select WMT19 test sentences that contain multiple rare/unseen characters.
    Save as a supplementary test set for rare-character analysis.
    """
    from datasets import load_dataset
    import random

    print(f"Building unseen-character test set (threshold < {rare_threshold} occurrences)...")
    dataset = load_dataset('wmt19', 'zh-en', split='train')

    # Build frequency map on the full training set
    freq: collections.Counter = collections.Counter()
    for ex in dataset:
        for ch in ex['translation']['zh']:
            freq[ch] += 1

    # Select sentences with >= min_rare_chars rare characters
    candidates = []
    for ex in dataset:
        zh = ex['translation']['zh']
        cjk = [c for c in zh if '一' <= c <= '鿿']
        n_rare = sum(1 for c in cjk if freq[c] < rare_threshold)
        if n_rare >= min_rare_chars:
            candidates.append({'chinese': zh, 'english': ex['translation']['en'],
                                'n_rare_chars': n_rare})

    random.seed(42)
    selected = random.sample(candidates, min(n_samples, len(candidates)))
    df = pd.DataFrame(selected)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved {len(df)} unseen-character sentences to {out_csv}")
    print(f"  Rare-char distribution: mean={df['n_rare_chars'].mean():.1f}, "
          f"max={df['n_rare_chars'].max()}")


# ── Main analysis ────────────────────────────────────────────────────────────

def run_analysis(
    preds_dir: str,
    hkccpn_csv: Optional[str] = None,
    freq_map: Optional[Dict[str, int]] = None,
    out_csv: str = "results/stratified_analysis.csv",
) -> pd.DataFrame:
    import glob

    pred_files = glob.glob(os.path.join(preds_dir, "*_preds.json"))
    if not pred_files:
        print(f"No prediction files found in {preds_dir}.")
        print("Run finetune_experiment.py with --save_predictions first.")
        return pd.DataFrame()

    hkccpn = load_hkccpn(hkccpn_csv) if hkccpn_csv and os.path.exists(hkccpn_csv) else None
    if hkccpn is not None:
        print(f"Using HKCCPN transparency norms ({len(hkccpn)} characters).")
    else:
        print("HKCCPN not available — using frequency-based stratification.")

    all_records = []

    for fpath in sorted(pred_files):
        name = Path(fpath).stem.replace("_preds", "")
        with open(fpath) as f:
            data = json.load(f)

        sources     = data.get("sources", [])
        predictions = data["predictions"]
        references  = data["references"]
        comet_scores = data.get("comet_scores")

        strata = stratify_sentences_by_transparency(
            sources, hkccpn=hkccpn, freq_map=freq_map
        )
        unseen_mask = unseen_character_mask(sources, freq_map or {}) if freq_map else None

        stratified = compute_stratified_metrics(predictions, references, strata, comet_scores)

        # Parse condition from filename
        parts = name.split("_")
        seed_part = [p for p in parts if p.startswith("s") and p[1:].isdigit()]
        if not seed_part:
            continue
        seed = int(seed_part[0][1:])
        idx_seed = parts.index(seed_part[0])
        train_size = int(parts[idx_seed - 1])
        rep = parts[idx_seed - 2]
        model = "_".join(parts[:idx_seed - 2])

        for _, row in stratified.iterrows():
            all_records.append({
                "model": model, "representation": rep,
                "train_size": train_size, "seed": seed,
                **row.to_dict(),
            })

        # Unseen-character sub-analysis
        if unseen_mask:
            for label, mask in [("unseen", unseen_mask),
                                  ("seen", [not m for m in unseen_mask])]:
                idx = [i for i, m in enumerate(mask) if m]
                if not idx:
                    continue
                preds_s = [predictions[i] for i in idx]
                refs_s  = [references[i]  for i in idx]
                from sacrebleu import corpus_bleu
                from sacrebleu.metrics import CHRF
                bleu = corpus_bleu(preds_s, [[r] for r in refs_s]).score
                chrf = CHRF().corpus_score(preds_s, [[r] for r in refs_s]).score
                all_records.append({
                    "model": model, "representation": rep,
                    "train_size": train_size, "seed": seed,
                    "stratum": label, "n": len(idx),
                    "bleu": round(bleu, 2), "chrf": round(chrf, 2), "comet": None,
                })

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Stratified analysis saved to {out_csv}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_dir", default="results/predictions/")
    parser.add_argument("--hkccpn_csv", default=None,
                        help="Path to HKCCPN norms CSV (Su, Yum & Lau 2022)")
    parser.add_argument("--out", default="results/stratified_analysis.csv")
    parser.add_argument("--build_unseen_test", action="store_true",
                        help="Build unseen-character test CSV and exit")
    parser.add_argument("--unseen_out", default="data/unseen_char_test.csv")
    args = parser.parse_args()

    if args.build_unseen_test:
        build_unseen_test_set(args.unseen_out)
        return

    freq_map = build_char_frequency_map("")
    df = run_analysis(
        preds_dir=args.preds_dir,
        hkccpn_csv=args.hkccpn_csv,
        freq_map=freq_map,
        out_csv=args.out,
    )

    if not df.empty:
        print("\nStratified results preview:")
        print(df.groupby(["model", "representation", "stratum"])
              .agg(bleu_mean=("bleu", "mean"), chrf_mean=("chrf", "mean"), n=("n", "sum"))
              .round(2).to_string())


if __name__ == "__main__":
    main()
