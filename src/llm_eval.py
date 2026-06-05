"""
llm_eval.py — LLM-as-judge pairwise translation evaluation.

Uses a locally cached instruction-tuned model (Llama 3.1 8B Instruct by default)
to make pairwise translation quality judgments, providing an independent evaluation
signal alongside COMET and corpus BLEU.

Task: given a Chinese source, an English reference, and two candidate translations
(A = baseline, B = alternative representation), judge which better preserves meaning
and which is more fluent. The judge only reads English — adequacy is assessed via
comparison to the reference, not direct Chinese comprehension.

Key comparison: baseline vs. morphemes (the paper's main claim).
Secondary: baseline vs. sentencepiece.

Usage:
    PYTHONPATH=src python src/llm_eval.py \
        --preds_dir results/predictions_v3/ \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --n_sentences 100 \
        --train_size 500
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

JUDGE_SYSTEM = """You are an expert evaluator of machine translation quality.
You will be shown a Chinese source sentence, an English reference translation, and two candidate English translations (A and B). Your job is to judge which translation is better.

Evaluate on two dimensions:
- Adequacy: how well the translation preserves the meaning of the source (use the reference as a guide to the intended meaning)
- Fluency: how natural and grammatical the English is

Respond ONLY in this exact format (no other text):
A_adequacy: [1-5]
A_fluency: [1-5]
B_adequacy: [1-5]
B_fluency: [1-5]
Winner: [A/B/Equal]
Reason: [one sentence]"""

JUDGE_TEMPLATE = """Source (Chinese): {source}
Reference: {reference}
Translation A: {trans_a}
Translation B: {trans_b}"""


def load_predictions(preds_dir: str, model_key: str, rep: str, train_size: int) -> dict:
    """Pool predictions across all seeds for a given condition."""
    pattern = os.path.join(preds_dir,
                           f"{model_key}_{rep}_{train_size}_s*_preds.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    pooled = defaultdict(list)
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        for k in ('sources', 'predictions', 'references'):
            pooled[k].extend(d.get(k, []))
    return dict(pooled)


def parse_judgment(text: str) -> dict:
    result = {'a_adequacy': None, 'a_fluency': None,
              'b_adequacy': None, 'b_fluency': None,
              'winner': None, 'reason': None, 'raw': text}
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('A_adequacy:'):
            try: result['a_adequacy'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('A_fluency:'):
            try: result['a_fluency'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('B_adequacy:'):
            try: result['b_adequacy'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('B_fluency:'):
            try: result['b_fluency'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('Winner:'):
            w = line.split(':', 1)[1].strip().upper()
            if w in ('A', 'B', 'EQUAL'):
                result['winner'] = w
        elif line.startswith('Reason:'):
            result['reason'] = line.split(':', 1)[1].strip()
    return result


def run_llm_eval(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

    print(f"Loading {args.model_id}  (4bit={args.load_in_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    load_kwargs = dict(device_map='auto')
    if args.load_in_4bit:
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,
                    max_new_tokens=128, do_sample=False, temperature=None,
                    top_p=None)
    print(f"✓ Model loaded  (GPU: {torch.cuda.is_available()})\n")

    all_results = []

    for mt_model in ['opus-mt', 'nllb-600M']:
        # Load baseline predictions
        base_data = load_predictions(args.preds_dir, mt_model, 'baseline', args.train_size)
        if not base_data:
            print(f"  No baseline predictions for {mt_model} n={args.train_size}, skipping")
            continue

        for rep_b in ['morphemes', 'sentencepiece']:
            rep_data = load_predictions(args.preds_dir, mt_model, rep_b, args.train_size)
            if not rep_data:
                print(f"  No {rep_b} predictions for {mt_model}, skipping")
                continue

            # Align by (source, reference) — sample deterministically
            n = min(len(base_data['sources']), len(rep_data['sources']))
            rng = random.Random(42)
            indices = rng.sample(range(n), min(args.n_sentences, n))

            print(f"\n{'='*60}")
            print(f"{mt_model} | baseline vs {rep_b} | n={args.train_size} | "
                  f"{len(indices)} sentences")
            print(f"{'='*60}")

            winners = []
            for idx in indices:
                src  = base_data['sources'][idx]
                ref  = base_data['references'][idx]
                ta   = base_data['predictions'][idx]   # A = baseline
                tb   = rep_data['predictions'][idx]    # B = rep

                prompt_user = JUDGE_TEMPLATE.format(
                    source=src, reference=ref, trans_a=ta, trans_b=tb)

                messages = [
                    {'role': 'system', 'content': JUDGE_SYSTEM},
                    {'role': 'user',   'content': prompt_user},
                ]
                try:
                    out = pipe(messages, pad_token_id=tokenizer.eos_token_id)
                    response = out[0]['generated_text'][-1]['content']
                except Exception as e:
                    print(f"  Error at idx {idx}: {e}")
                    response = ''

                judgment = parse_judgment(response)
                judgment.update({'mt_model': mt_model, 'rep_b': rep_b,
                                 'train_size': args.train_size, 'source_idx': idx,
                                 'source': src, 'reference': ref,
                                 'trans_a': ta, 'trans_b': tb})
                all_results.append(judgment)
                if judgment['winner']:
                    winners.append(judgment['winner'])
                    if len(winners) % 10 == 0:
                        c = Counter(winners)
                        print(f"  [{len(winners):3d}/{len(indices)}] "
                              f"A(baseline)={c['A']}  B({rep_b})={c['B']}  "
                              f"Equal={c['EQUAL']}")

            if winners:
                c = Counter(winners)
                total = len(winners)
                print(f"\n  FINAL: A(baseline)={c['A']} ({c['A']/total:.0%})  "
                      f"B({rep_b})={c['B']} ({c['B']/total:.0%})  "
                      f"Equal={c['EQUAL']} ({c['EQUAL']/total:.0%})")
                # Mean scores
                a_adeq = np.mean([r['a_adequacy'] for r in all_results
                                  if r.get('mt_model') == mt_model
                                  and r.get('rep_b') == rep_b
                                  and r.get('a_adequacy') is not None])
                b_adeq = np.mean([r['b_adequacy'] for r in all_results
                                  if r.get('mt_model') == mt_model
                                  and r.get('rep_b') == rep_b
                                  and r.get('b_adequacy') is not None])
                print(f"  Mean adequacy: A(baseline)={a_adeq:.2f}  B({rep_b})={b_adeq:.2f}")

    # Save
    out_dir = PROJECT_DIR / 'results' / 'llm_eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / 'judgments.csv', index=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summ = df.groupby(['mt_model', 'rep_b', 'winner']).size().unstack(fill_value=0)
    print(summ.to_string())
    print(f"\n✓ Saved {len(df)} judgments to {out_dir}/judgments.csv")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir',   default='results/predictions_v3/')
    parser.add_argument('--model_id',    default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--n_sentences', type=int, default=100)
    parser.add_argument('--train_size',  type=int, default=500)
    parser.add_argument('--load_in_4bit', action='store_true')
    args = parser.parse_args()
    run_llm_eval(args)


if __name__ == '__main__':
    main()
