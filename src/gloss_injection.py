"""
gloss_injection.py — Character semantic gloss injection experiment.

Tests the original project intuition: does explicitly providing the meaning of
rare/unseen Chinese characters improve translation quality?

Operationalization of "how humans learn characters":
  - Identify rare characters in each source sentence
  - Prepend brief English glosses from CC-CEDICT: "[稀: rare; 澄: clear]"
  - For semantically transparent characters (HKCCPN transparency > threshold),
    also include the radical composition: "[明 (日+月): bright]"
  - Run zero-shot inference; compare glossed vs. unglossed on BLEU/chrF/COMET
  - Stratify by semantic transparency to test whether glosses help more
    for transparent characters (the theoretically motivated hypothesis)

Usage:
    PYTHONPATH=src python src/gloss_injection.py
    (or as SLURM job via scripts/run_gloss_injection.sh)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

# ── CC-CEDICT loader ──────────────────────────────────────────────────────────

def load_cedict(path: str) -> dict:
    """
    Parse CC-CEDICT into a char → first English definition mapping.
    Only keeps single-character entries to get character-level glosses.
    """
    glosses = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            # Format: Traditional Simplified [pinyin] /def1/def2/
            parts = line.strip().split(' ', 2)
            if len(parts) < 3:
                continue
            simp = parts[1]
            if len(simp) != 1:  # single characters only
                continue
            # Extract first definition
            defs = parts[2]
            start = defs.find('/')
            end   = defs.find('/', start + 1)
            if start >= 0 and end > start:
                first_def = defs[start+1:end].strip()
                # Skip pinyin-only entries and classifier entries
                if not first_def.startswith('CL:') and len(first_def) < 60:
                    glosses[simp] = first_def
    print(f"  CC-CEDICT: {len(glosses)} single-character entries")
    return glosses


def load_hkccpn(path: str) -> dict:
    """Load HKCCPN transparency scores (Traditional + Simplified via opencc)."""
    try:
        df = pd.read_excel(path)
        try:
            import opencc
            conv = opencc.OpenCC('t2s')
            simp = {conv.convert(r['Character']): r['SemanticRadicalTrans']
                    for _, r in df.iterrows() if pd.notna(r.get('SemanticRadicalTrans'))}
        except Exception:
            simp = {}
        trad = {r['Character']: r['SemanticRadicalTrans']
                for _, r in df.iterrows() if pd.notna(r.get('SemanticRadicalTrans'))}
        combined = {**trad, **simp}
        # Normalize 0-1
        vals = list(combined.values())
        lo, hi = min(vals), max(vals)
        if hi > lo:
            combined = {k: (v - lo) / (hi - lo) for k, v in combined.items()}
        print(f"  HKCCPN: {len(combined)} characters")
        return combined
    except Exception as e:
        print(f"  HKCCPN unavailable: {e}")
        return {}


# ── Gloss building ────────────────────────────────────────────────────────────

def build_gloss(
    sentence: str,
    rare_chars: set,
    cedict: dict,
    ids_map: dict,
    transparency: dict,
    transparency_threshold: float = 0.6,
    max_glosses: int = 5,
) -> str:
    """
    Build a gloss prefix for rare characters in a sentence.

    Format: "[明 (日+月): bright; 稀: rare, scarce]"

    For transparent characters, include radical composition.
    Capped at max_glosses to avoid overwhelming the model.
    """
    layout = set('⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻')
    seen = set()
    glosses = []

    for ch in sentence:
        if ch not in rare_chars or ch in seen:
            continue
        seen.add(ch)

        defn = cedict.get(ch)
        if not defn:
            continue

        # Shorten definition to first meaningful phrase
        short_def = defn.split(';')[0].split(',')[0].strip()
        if len(short_def) > 30:
            short_def = short_def[:30]

        # Add radical composition for transparent characters
        trans_score = transparency.get(ch, 0)
        if trans_score >= transparency_threshold and ch in ids_map:
            raw = ids_map[ch]
            components = [c for c in raw if c not in layout][:3]
            if components and components != [ch]:
                comp_str = '+'.join(components)
                glosses.append(f"{ch} ({comp_str}): {short_def}")
            else:
                glosses.append(f"{ch}: {short_def}")
        else:
            glosses.append(f"{ch}: {short_def}")

        if len(glosses) >= max_glosses:
            break

    if not glosses:
        return sentence
    return f"[{'; '.join(glosses)}] {sentence}"


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, texts, model_key, batch_size=16, max_length=256):
    import torch
    MODEL_CONFIGS = {
        'opus-mt': {'family': 'marian', 'tgt_lang': None},
        'nllb-600M': {'family': 'nllb', 'tgt_lang': 'eng_Latn',
                      'tgt_id': tokenizer.convert_tokens_to_ids('eng_Latn')
                      if hasattr(tokenizer, 'convert_tokens_to_ids') else None},
    }
    cfg = MODEL_CONFIGS[model_key]
    gen_kw = {'max_length': max_length}
    if cfg['family'] == 'nllb' and cfg.get('tgt_id'):
        gen_kw['forced_bos_token_id'] = cfg['tgt_id']

    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        preds.extend(tokenizer.decode(t, skip_special_tokens=True) for t in out)
    return preds


def evaluate(preds, refs, srcs=None, comet_model=None):
    from sacrebleu import corpus_bleu, sentence_bleu
    from sacrebleu.metrics import CHRF
    sent_bleus = [sentence_bleu(p, [r]).score for p, r in zip(preds, refs)]
    res = {
        'bleu':     corpus_bleu(preds, [[r] for r in refs]).score,
        'chrf':     CHRF().corpus_score(preds, [[r] for r in refs]).score,
        'sent_bleus': sent_bleus,
        'comet': None,
    }
    if srcs and comet_model:
        try:
            out = comet_model.predict(
                [{'src': s, 'mt': p, 'ref': r} for s, p, r in zip(srcs, preds, refs)],
                batch_size=32, gpus=1)
            res['comet'] = float(np.mean(out.scores))
            res['comet_scores'] = [float(x) for x in out.scores]
        except Exception as e:
            print(f"  COMET error: {e}")
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print("=" * 70)
    print("GLOSS INJECTION EXPERIMENT")
    print("=" * 70)

    # ── Load resources
    cedict_path = str(PROJECT_DIR / 'data' / 'cedict_ts.u8')
    hkccpn_path = str(PROJECT_DIR / 'data' / 'HK_RatingsNorm_2022.xlsx')
    ids_path    = str(PROJECT_DIR / 'data' / 'ids.txt')
    unseen_path = str(PROJECT_DIR / 'data' / 'unseen_char_test.csv')

    print("Loading resources...")
    cedict = load_cedict(cedict_path) if os.path.exists(cedict_path) else {}
    hkccpn = load_hkccpn(hkccpn_path) if os.path.exists(hkccpn_path) else {}

    # Load IDS map
    ids_map = {}
    if os.path.exists(ids_path):
        with open(ids_path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ids_map[parts[1]] = parts[2]
    print(f"  IDS map: {len(ids_map)} entries")

    # Load unseen-char test set
    unseen_df = pd.read_csv(unseen_path)
    print(f"  Unseen-char test: {len(unseen_df)} sentences\n")

    # Build WMT19 frequency map for rare-char detection
    print("Building character frequency map...")
    from collections import Counter
    freq: Counter = Counter()
    try:
        from datasets import load_dataset
        ds = load_dataset('wmt19', 'zh-en', split='train')
        for ex in ds:
            for ch in ex['translation']['zh']:
                freq[ch] += 1
        print(f"  {len(freq)} unique chars\n")
    except Exception as e:
        print(f"  Warning: {e}\n")

    rare_threshold = 10
    COMET_MODEL = None
    try:
        from comet import download_model, load_from_checkpoint
        COMET_MODEL = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
        print("✓ COMET loaded\n")
    except Exception as e:
        print(f"⚠ COMET unavailable: {e}\n")

    results = []

    for model_key, model_id, src_lang in [
        ('opus-mt',   'Helsinki-NLP/opus-mt-zh-en',           None),
        ('nllb-600M', 'facebook/nllb-200-distilled-600M', 'zho_Hans'),
    ]:
        print(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")

        from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
        if model_key == 'opus-mt':
            tokenizer = MarianTokenizer.from_pretrained(model_id)
            model     = MarianMTModel.from_pretrained(model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)
            model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        if torch.cuda.is_available():
            model = model.cuda()

        sources = unseen_df['chinese'].tolist()
        refs    = unseen_df['english'].tolist()

        # Identify rare chars per sentence
        rare_sets = [
            {ch for ch in s if '一' <= ch <= '鿿' and freq.get(ch, 0) < rare_threshold}
            for s in sources
        ]

        # Build three conditions
        conditions = {
            'unglossed': sources,
            'glossed_all': [
                build_gloss(s, r, cedict, ids_map, hkccpn, transparency_threshold=0.0)
                for s, r in zip(sources, rare_sets)
            ],
            'glossed_transparent_only': [
                build_gloss(s, r, cedict, ids_map, hkccpn, transparency_threshold=0.6)
                for s, r in zip(sources, rare_sets)
            ],
        }

        # Show a sample
        print(f"\nSample input:")
        print(f"  unglossed:             {conditions['unglossed'][0][:80]}")
        print(f"  glossed_all:           {conditions['glossed_all'][0][:100]}")
        print(f"  glossed_transparent:   {conditions['glossed_transparent_only'][0][:100]}")

        for cond_name, inputs in conditions.items():
            preds  = run_inference(model, tokenizer, inputs, model_key)
            scores = evaluate(preds, refs,
                              srcs=sources if COMET_MODEL else None,
                              comet_model=COMET_MODEL)

            print(f"\n  [{cond_name}]")
            print(f"    BLEU={scores['bleu']:.2f}  chrF={scores['chrf']:.2f}"
                  + (f"  COMET={scores['comet']:.4f}" if scores['comet'] else ""))

            # Stratify by transparency
            trans_scores = [
                np.mean([hkccpn.get(ch, 0) for ch in s if '一' <= ch <= '鿿'])
                for s in sources
            ]
            med = np.median(trans_scores)
            for stratum, idx in [
                ('high_trans', [i for i, t in enumerate(trans_scores) if t >= med]),
                ('low_trans',  [i for i, t in enumerate(trans_scores) if t < med]),
            ]:
                if not idx: continue
                from sacrebleu import corpus_bleu
                from sacrebleu.metrics import CHRF
                s_bleu = corpus_bleu([preds[i] for i in idx],
                                     [[refs[i]] for i in idx]).score
                s_chrf = CHRF().corpus_score([preds[i] for i in idx],
                                              [[refs[i]] for i in idx]).score
                s_comet = (float(np.mean([scores['comet_scores'][i] for i in idx]))
                           if scores.get('comet_scores') else None)
                print(f"      {stratum:15s}: BLEU={s_bleu:.2f}  chrF={s_chrf:.2f}"
                      + (f"  COMET={s_comet:.4f}" if s_comet else ""))
                results.append(dict(model=model_key, condition=cond_name, stratum=stratum,
                                    bleu=s_bleu, chrf=s_chrf, comet=s_comet, n=len(idx)))

            results.append(dict(model=model_key, condition=cond_name, stratum='all',
                                bleu=scores['bleu'], chrf=scores['chrf'],
                                comet=scores['comet'], n=len(sources)))

        # Save predictions
        preds_dir = PROJECT_DIR / 'results' / 'gloss_injection'
        preds_dir.mkdir(parents=True, exist_ok=True)
        for cond_name, inputs in conditions.items():
            preds = run_inference(model, tokenizer, inputs, model_key)
            with open(preds_dir / f'{model_key}_{cond_name}_preds.json', 'w') as f:
                json.dump({'sources': sources, 'glossed_inputs': inputs,
                           'predictions': preds, 'references': refs}, f,
                          ensure_ascii=False)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    df = pd.DataFrame(results)
    out = PROJECT_DIR / 'results' / 'gloss_injection_results.csv'
    df.to_csv(out, index=False)
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(df[df['stratum'] == 'all'].pivot_table(
        index=['model', 'condition'],
        values=['bleu', 'chrf', 'comet'],
    ).round(3).to_string())
    print(f"\n✓ Saved to {out}")


if __name__ == '__main__':
    main()
