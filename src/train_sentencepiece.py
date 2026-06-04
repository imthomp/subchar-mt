"""
train_sentencepiece.py — Train a SentencePiece unigram model on WMT19 Chinese text.

Run this ONCE on a login node (needs internet or cached HF data) before submitting
the v2 experiment. The trained model is saved to data/zh_sp.model and loaded by
LinguisticEncoder in finetune_experiment.py.

Usage:
    python src/train_sentencepiece.py
    python src/train_sentencepiece.py --vocab_size 8000 --input_sentence_size 50000

The SentencePiece 'sentencepiece' representation serves as a data-driven subword
baseline against jieba's dictionary-based morpheme segmentation. Per Si et al.
(TACL 2023), purely statistical segmentation often matches linguistically-motivated
sub-character encodings — including this control tests whether jieba's segmentation
advantage is linguistic or merely structural.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"


def train(vocab_size: int = 8000, input_sentence_size: int = 100000) -> None:
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed.")
        sys.exit(1)

    model_path = DATA_DIR / "zh_sp.model"
    if model_path.exists():
        print(f"Model already exists at {model_path}. Delete it to retrain.")
        return

    print(f"Loading WMT19 zh-en (up to {input_sentence_size} sentences)...")
    dataset = load_dataset("wmt19", "zh-en", split="train")
    n = min(input_sentence_size, len(dataset))
    chinese_sentences = [dataset[i]["translation"]["zh"] for i in range(n)]
    print(f"  Loaded {len(chinese_sentences)} sentences.")

    # Write to a temporary plain-text file (SP trainer requires a file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     encoding='utf-8', delete=False) as f:
        tmp_path = f.name
        for sent in chinese_sentences:
            f.write(sent.strip() + "\n")

    print(f"Training SentencePiece unigram model (vocab_size={vocab_size})...")
    prefix = str(DATA_DIR / "zh_sp")
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,   # high coverage for CJK
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
    )
    os.unlink(tmp_path)

    print(f"✓ Model saved to {prefix}.model and {prefix}.vocab")

    # Quick sanity check
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{prefix}.model")
    sample = "中国人民解放军"
    pieces = sp.EncodeAsPieces(sample)
    print(f"\nSanity check: '{sample}' → {pieces}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--input_sentence_size", type=int, default=100000)
    args = parser.parse_args()
    train(vocab_size=args.vocab_size, input_sentence_size=args.input_sentence_size)


if __name__ == "__main__":
    main()
