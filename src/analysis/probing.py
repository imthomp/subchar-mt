"""
probing.py — Probing classifiers on encoder hidden states per representation.

Following Belinkov et al. (Computational Linguistics 2020), we train lightweight
linear probes on the encoder's final hidden states to ask: what linguistic
information does each representation encode?

Probes:
  - POS tagging (morphosyntactic signal)
  - Named-entity recognition (lexical-semantic signal)
  - Character frequency bin (does the representation encode familiarity?)
  - Semantic-radical identity (does radical decomposition preserve radical signal?)

This is the "mechanistic evidence" that converts correlation into explanation.
A probe that works well for morphemes but not radicals on semantic tasks would
directly support the pretraining-mismatch hypothesis.

Usage:
    # Extract hidden states (run after fine-tuning with --save_predictions):
    python src/analysis/probing.py extract \\
        --model_dir models_supercomputer/opus-mt_morphemes_100_s42 \\
        --representation morphemes \\
        --data_csv finetuned_results_ALL.csv \\
        --out results/probing/hidden_states_morphemes.pt

    # Train and evaluate probes:
    python src/analysis/probing.py probe \\
        --states_dir results/probing/ \\
        --task pos \\
        --out results/probing/probe_results.csv

Prerequisites:
    pip install sklearn jieba stanza  (stanza for POS/NER on Chinese)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import stanza  # Stanford NLP pipeline for Chinese POS/NER
except ImportError:
    stanza = None  # type: ignore


# ── Hidden state extraction ───────────────────────────────────────────────────

def extract_hidden_states(
    model_dir: str,
    representation: str,
    source_sentences: List[str],
    model_key: str,
    out_path: str,
    layer: int = -1,          # which encoder layer; -1 = last
    pool: str = 'mean',       # 'mean' | 'cls' | 'last'
    batch_size: int = 16,
) -> None:
    """
    Forward-pass source_sentences through the encoder and save pooled hidden states.
    States are saved as a dict: {'states': np.array [n × d], 'sources': [...]}
    """
    if torch is None:
        raise ImportError("PyTorch required for hidden state extraction.")

    from encoder import LinguisticEncoder
    from finetune_experiment import MODEL_CONFIGS, _make_encoder

    # Apply representation preprocessing
    enc = _make_encoder()
    if representation == 'baseline':
        texts = source_sentences
    elif representation == 'morphemes':
        texts = [enc.segment_morphemes(t) for t in source_sentences]
    elif representation == 'pinyin':
        texts = [enc.to_pinyin(t) for t in source_sentences]
    elif representation == 'radicals':
        texts = [enc.to_radicals(t) for t in source_sentences]
    elif representation == 'byte':
        texts = [enc.to_bytes(t) for t in source_sentences]
    elif representation == 'random_index':
        texts = [enc.to_random_index(t) for t in source_sentences]
    elif representation == 'sentencepiece':
        texts = [enc.to_sentencepiece(t) for t in source_sentences]
    else:
        texts = [' '.join(enc.to_wubi(t) if 'wubi' in representation
                          else enc.to_cangjie(t))
                 for t in source_sentences]

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, output_hidden_states=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    all_states = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.encoder(**inputs, output_hidden_states=True)

        # out.hidden_states: tuple of (n_layers+1) tensors [batch × seq × d]
        hs = out.hidden_states[layer]  # [batch × seq × d]
        attn_mask = inputs['attention_mask'].unsqueeze(-1).float()

        if pool == 'mean':
            pooled = (hs * attn_mask).sum(1) / attn_mask.sum(1)
        elif pool == 'cls':
            pooled = hs[:, 0, :]
        else:
            # Last non-padding token
            lengths = inputs['attention_mask'].sum(1) - 1
            pooled = hs[torch.arange(len(batch)), lengths]

        all_states.append(pooled.cpu().numpy())

    states = np.vstack(all_states)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             states=states,
             sources=np.array(source_sentences, dtype=object))
    print(f"✓ Saved {states.shape} hidden states to {out_path}")


# ── Label generation ──────────────────────────────────────────────────────────

def get_pos_labels(sentences: List[str]) -> List[Optional[str]]:
    """Get majority POS tag per sentence using Stanza Chinese pipeline."""
    if stanza is None:
        print("Warning: stanza not installed. Install with: pip install stanza")
        print("  Then: import stanza; stanza.download('zh')")
        return [None] * len(sentences)

    try:
        nlp = stanza.Pipeline('zh', processors='tokenize,pos', use_gpu=False, verbose=False)
    except Exception as e:
        print(f"Warning: Stanza pipeline failed: {e}")
        return [None] * len(sentences)

    labels = []
    for sent in sentences:
        try:
            doc = nlp(sent)
            # Majority POS across all tokens
            pos_counts: Dict[str, int] = {}
            for s in doc.sentences:
                for word in s.words:
                    pos_counts[word.upos] = pos_counts.get(word.upos, 0) + 1
            labels.append(max(pos_counts, key=pos_counts.get) if pos_counts else None)
        except Exception:
            labels.append(None)
    return labels


def get_frequency_labels(
    sentences: List[str],
    freq_map: Dict[str, int],
    n_bins: int = 3,
) -> List[Optional[int]]:
    """Bin each sentence by mean character log-frequency."""
    scores = []
    for sent in sentences:
        cjk = [c for c in sent if '一' <= c <= '鿿']
        if cjk:
            scores.append(np.mean([np.log1p(freq_map.get(c, 0)) for c in cjk]))
        else:
            scores.append(None)

    finite = [s for s in scores if s is not None]
    if not finite:
        return [None] * len(sentences)

    thresholds = np.percentile(finite, np.linspace(0, 100, n_bins + 1)[1:-1])
    labels = []
    for s in scores:
        if s is None:
            labels.append(None)
        else:
            labels.append(int(np.searchsorted(thresholds, s)))
    return labels


# ── Linear probe ──────────────────────────────────────────────────────────────

def train_probe(
    states: np.ndarray,
    labels: List,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Train and evaluate a linear probe (logistic regression) on encoder states.
    Returns accuracy, macro-F1, and a feature importance vector.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Filter out None labels
    valid_idx = [i for i, l in enumerate(labels) if l is not None]
    if len(valid_idx) < 10:
        return {'error': 'insufficient valid labels', 'n': len(valid_idx)}

    X = states[valid_idx]
    y_raw = [labels[i] for i in valid_idx]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Majority-class baseline
    from collections import Counter
    majority = max(Counter(y_train).values()) / len(y_train)

    return {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': len(le.classes_),
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(f1), 4),
        'majority_baseline': round(float(majority), 4),
        'delta_over_majority': round(float(acc - majority), 4),
        'classes': list(le.classes_),
    }


# ── Cross-representation probe comparison ────────────────────────────────────

def compare_representations(
    states_dir: str,
    task: str,
    freq_map: Optional[Dict] = None,
    out_csv: str = "results/probing/probe_results.csv",
) -> pd.DataFrame:
    """
    Load all hidden-state files from states_dir and run probes for the given task.
    Compares probe accuracy across representations — the key mechanistic result.
    """
    import glob

    state_files = glob.glob(os.path.join(states_dir, "*.npz"))
    if not state_files:
        print(f"No .npz state files found in {states_dir}")
        return pd.DataFrame()

    records = []
    for fpath in sorted(state_files):
        name = Path(fpath).stem
        data = np.load(fpath, allow_pickle=True)
        states = data['states']
        sources = list(data['sources'])

        # Parse representation from filename (format: {model}_{rep}_{train_size}_s{seed})
        parts = name.split("_")
        rep = parts[1] if len(parts) > 1 else name

        print(f"\n  [{name}] task={task} n={len(states)}")

        if task == 'pos':
            labels = get_pos_labels(sources)
        elif task == 'frequency':
            labels = get_frequency_labels(sources, freq_map or {})
        else:
            print(f"  Unknown task: {task}")
            continue

        result = train_probe(states, labels)
        result['file'] = name
        result['representation'] = rep
        result['task'] = task
        records.append(result)
        print(f"    acc={result.get('accuracy', '?')} "
              f"f1={result.get('macro_f1', '?')} "
              f"majority={result.get('majority_baseline', '?')}")

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Probe results saved to {out_csv}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command')

    ext = sub.add_parser('extract', help='Extract encoder hidden states')
    ext.add_argument('--model_dir', required=True)
    ext.add_argument('--representation', required=True)
    ext.add_argument('--source_file', default=None,
                     help='Plain-text file with one Chinese sentence per line')
    ext.add_argument('--model_key', default='opus-mt')
    ext.add_argument('--out', required=True)
    ext.add_argument('--layer', type=int, default=-1)
    ext.add_argument('--pool', choices=['mean', 'cls', 'last'], default='mean')
    ext.add_argument('--n', type=int, default=500)

    prb = sub.add_parser('probe', help='Train probing classifiers')
    prb.add_argument('--states_dir', default='results/probing/')
    prb.add_argument('--task', choices=['pos', 'frequency'], default='pos')
    prb.add_argument('--out', default='results/probing/probe_results.csv')

    args = parser.parse_args()

    if args.command == 'extract':
        if args.source_file:
            with open(args.source_file) as f:
                sources = [l.strip() for l in f if l.strip()][:args.n]
        else:
            # Load from WMT19 test set
            from datasets import load_dataset
            dataset = load_dataset('wmt19', 'zh-en', split='train')
            sources = [dataset[i]['translation']['zh'] for i in range(args.n)]
        extract_hidden_states(
            model_dir=args.model_dir,
            representation=args.representation,
            source_sentences=sources,
            model_key=args.model_key,
            out_path=args.out,
            layer=args.layer,
            pool=args.pool,
        )

    elif args.command == 'probe':
        compare_representations(
            states_dir=args.states_dir,
            task=args.task,
            out_csv=args.out,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
