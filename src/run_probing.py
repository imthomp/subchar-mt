"""
run_probing.py — Train probing classifiers on v3 encoder hidden states.

Uses frequency-bin probing (no stanza required) and POS probing (requires stanza).
Compares how well each representation's encoder states predict:
  1. Character frequency bin (low/mid/high) — tests if representation preserves familiarity signal
  2. POS of dominant token in sentence — tests morphosyntactic signal

Run on login node (CPU only, ~5 min for all 120 state files).

Usage:
    PYTHONPATH=src python src/run_probing.py
"""
import collections
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))


def build_freq_map(n=50000):
    """Build character frequency map from WMT19 (first n examples)."""
    try:
        os.environ.setdefault('HF_DATASETS_OFFLINE', '0')
        from datasets import load_dataset
        ds = load_dataset('wmt19', 'zh-en', split='train')
        counter: collections.Counter = collections.Counter()
        for i, ex in enumerate(ds):
            if i >= n: break
            for ch in ex['translation']['zh']:
                counter[ch] += 1
        print(f"  Frequency map: {len(counter)} unique chars from {n} examples")
        return dict(counter)
    except Exception as e:
        print(f"  Warning: could not build freq map: {e}")
        return {}


def get_freq_labels(sources, freq_map, n_bins=3):
    scores = []
    for sent in sources:
        cjk = [c for c in str(sent) if '一' <= c <= '鿿']
        if cjk and freq_map:
            scores.append(np.mean([np.log1p(freq_map.get(c, 0)) for c in cjk]))
        else:
            scores.append(None)
    finite = [s for s in scores if s is not None]
    if not finite:
        return [None] * len(sources)
    thresholds = np.percentile(finite, np.linspace(0, 100, n_bins + 1)[1:-1])
    labels = []
    for s in scores:
        if s is None:
            labels.append(None)
        elif s <= thresholds[0]:
            labels.append('low')
        elif len(thresholds) > 1 and s <= thresholds[1]:
            labels.append('mid')
        else:
            labels.append('high')
    return labels


def get_pos_labels(sources):
    try:
        # Patch MD5 checks — blocked by FIPS mode on BYU cluster
        import stanza.resources.common as _stanza_common
        _stanza_common.get_md5 = lambda path: "skipped"
        _stanza_common.assert_file_exists = lambda path, md5=None, alternate_md5=None: None
        import stanza
        nlp = stanza.Pipeline('zh-hans', processors='tokenize,pos',
                               use_gpu=False, verbose=False,
                               download_method=None)
        labels = []
        for sent in sources:
            try:
                doc = nlp(str(sent))
                pos_counts: dict = {}
                for s in doc.sentences:
                    for word in s.words:
                        pos_counts[word.upos] = pos_counts.get(word.upos, 0) + 1
                labels.append(max(pos_counts, key=pos_counts.get) if pos_counts else None)
            except Exception:
                labels.append(None)
        return labels
    except Exception as e:
        print(f"  stanza unavailable: {e}")
        return None


def train_probe(states, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    valid = [(states[i], labels[i]) for i in range(len(labels)) if labels[i] is not None]
    if len(valid) < 20:
        return None
    X = np.array([v[0] for v in valid])
    y_raw = [v[1] for v in valid]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Need at least 2 classes
    if len(np.unique(y)) < 2:
        return None

    # Drop classes with < 2 members so stratified split doesn't fail
    from collections import Counter
    counts = Counter(y.tolist())
    keep = np.array([i for i in range(len(X)) if counts[y[i]] >= 2])
    if len(keep) < 20:
        return None
    X, y = X[keep], y[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    from collections import Counter
    majority = max(Counter(y_tr).values()) / len(y_tr)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='macro', zero_division=0)
    return {
        'n': len(valid), 'n_classes': len(le.classes_),
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(f1), 4),
        'majority_baseline': round(float(majority), 4),
        'delta_acc': round(float(acc - majority), 4),
    }


def parse_state_filename(stem):
    # e.g. nllb-600M_baseline_1000_s42_states  OR  opus-mt_radicals_250_s999_states
    stem = stem.replace('_states', '')
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


def main():
    states_dir = PROJECT_DIR / 'results' / 'probing'
    out_csv    = PROJECT_DIR / 'results' / 'probing' / 'probe_results.csv'

    state_files = sorted(glob.glob(str(states_dir / '*.npz')))
    print(f"Found {len(state_files)} state files")

    print("Building character frequency map...")
    freq_map = build_freq_map(n=50000)

    # Try to get stanza POS labels (requires stanza zh model downloaded)
    # We'll use the first file's sources as a test
    pos_available = False
    if state_files:
        d = np.load(state_files[0], allow_pickle=True)
        sources_test = list(d['sources'])[:5]
        pos_test = get_pos_labels(sources_test)
        pos_available = pos_test is not None and any(l is not None for l in pos_test)
    print(f"POS probing available: {pos_available}")

    records = []
    for fpath in state_files:
        stem = Path(fpath).stem
        meta = parse_state_filename(stem)
        if meta is None:
            continue

        data   = np.load(fpath, allow_pickle=True)
        states = data['states']
        sources = list(data['sources'])

        for task in (['frequency', 'pos'] if pos_available else ['frequency']):
            if task == 'frequency':
                labels = get_freq_labels(sources, freq_map)
            else:
                labels = get_pos_labels(sources)
                if labels is None:
                    continue

            result = train_probe(states, labels)
            if result is None:
                print(f"  [{stem}] task={task} — skipped (insufficient data)")
                continue

            row = {**meta, 'task': task, **result}
            records.append(row)
            print(f"  [{meta['model']:9s}|{meta['rep']:14s}|n={meta['train_size']:4d}|s={meta['seed']}]"
                  f"  {task:10s}  acc={result['accuracy']:.4f}  "
                  f"Δ={result['delta_acc']:+.4f}  f1={result['macro_f1']:.4f}")

    if not records:
        print("No results.")
        return

    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved {len(df)} probe results to {out_csv}")

    # Summary: mean delta_acc per representation × task
    print("\n" + "=" * 80)
    print("SUMMARY: mean probe accuracy delta over majority baseline")
    print("=" * 80)
    summ = df.groupby(['model', 'rep', 'task']).agg(
        delta_mean=('delta_acc', 'mean'),
        acc_mean=('accuracy', 'mean'),
        majority_mean=('majority_baseline', 'mean'),
        n_files=('accuracy', 'count'),
    ).round(4)
    print(summ.to_string())


if __name__ == '__main__':
    main()
