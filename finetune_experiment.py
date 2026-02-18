"""
Fine-tuning experiment for linguistically-informed low-resource MT
BYU Supercomputer submission script

Author: Isaac (PhD student)
Course: CS 501R
Date: 2026-01-30
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for better GPU utilization
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("="*80)
print("LINGUISTICALLY-INFORMED LOW-RESOURCE MT EXPERIMENT")
print("="*80)
print(f"Running on: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'}")
print(f"Working directory: {os.getcwd()}\n")

# Import libraries
from collections import Counter
from typing import List, Dict
import torch

from datasets import load_dataset, Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from sacrebleu import corpus_bleu, sentence_bleu
from bert_score import score as bert_score

from encoder import LinguisticEncoder

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

print("✓ LinguisticEncoder loaded\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(max_samples=2000, seed=42):
    """Load WMT19 Chinese-English dataset and create linguistic representations"""

    print(f"Loading WMT19 dataset (zh-en, {max_samples} samples)...")
    dataset = load_dataset('wmt19', 'zh-en', split='train')
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    df = pd.DataFrame(dataset)
    df['chinese'] = df['translation'].apply(lambda x: x['zh'])
    df['english'] = df['translation'].apply(lambda x: x['en'])

    print("Creating linguistic representations...")
    encoder = LinguisticEncoder()
    df['pinyin'] = df['chinese'].apply(lambda x: encoder.to_pinyin(x))
    df['pinyin_no_tone'] = df['chinese'].apply(lambda x: encoder.to_pinyin(x, with_tone=False))
    df['morphemes'] = df['chinese'].apply(lambda x: encoder.segment_morphemes(x))
    df['radicals'] = df['chinese'].apply(lambda x: encoder.to_radicals(x))

    print(f"✓ Prepared {len(df)} examples with linguistic representations\n")
    return df

def create_low_resource_splits(df, train_sizes=[100, 500, 1000], val_size=50, test_size=50, seed=42):
    """Create train/val/test splits for low-resource scenarios"""

    np.random.seed(seed)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    test_df = df_shuffled.iloc[:test_size].copy()
    val_df = df_shuffled.iloc[test_size:test_size+val_size].copy()

    datasets = {}
    remaining_start = test_size + val_size

    for train_size in sorted(train_sizes):
        if remaining_start + train_size > len(df_shuffled):
            print(f"Warning: Not enough data for train_size={train_size}")
            continue

        train_df = df_shuffled.iloc[remaining_start:remaining_start+train_size].copy()
        datasets[f'train_{train_size}'] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        print(f"✓ Created split: {train_size} train, {len(val_df)} val, {len(test_df)} test")

    return datasets, test_df, val_df

# Load data
df_full = load_and_prepare_data(max_samples=2000)
splits, test_data, val_data = create_low_resource_splits(df_full, train_sizes=[100, 500, 1000])

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_translations(predictions, references, metric='both'):
    """Evaluate translations using BLEU and/or BERTScore"""
    results = {}

    if metric in ['bleu', 'both']:
        sentence_bleus = []
        for pred, ref in zip(predictions, references):
            bleu = sentence_bleu(pred, [ref])
            sentence_bleus.append(bleu.score)

        corpus_bleu_score = corpus_bleu(predictions, [[r] for r in references])
        results['bleu_corpus'] = corpus_bleu_score.score
        results['bleu_sentence_mean'] = np.mean(sentence_bleus)
        results['bleu_sentence_std'] = np.std(sentence_bleus)

    if metric in ['bertscore', 'both']:
        P, R, F1 = bert_score(predictions, references, lang='en',
                             verbose=False, rescale_with_baseline=True, device='cuda')
        results['bertscore_precision'] = P.mean().item()
        results['bertscore_recall'] = R.mean().item()
        results['bertscore_f1'] = F1.mean().item()
        results['bertscore_f1_std'] = F1.std().item()

    return results

print("✓ Evaluation functions loaded\n")

# ============================================================================
# FINE-TUNING
# ============================================================================

class RepresentationFineTuner:
    """Fine-tune MT models on different linguistic representations"""

    def __init__(self, base_model='Helsinki-NLP/opus-mt-zh-en'):
        self.base_model = base_model
        self.encoder = LinguisticEncoder()

    def prepare_dataset(self, df, representation='baseline'):
        """Convert dataframe to HuggingFace Dataset"""
        data = []
        for _, row in df.iterrows():
            source = row['chinese']

            if representation == 'baseline':
                source_text = source
            elif representation == 'pinyin':
                source_text = self.encoder.to_pinyin(source)
            elif representation == 'pinyin_no_tone':
                source_text = self.encoder.to_pinyin(source, with_tone=False)
            elif representation == 'morphemes':
                source_text = self.encoder.segment_morphemes(source)
            elif representation == 'radicals':
                source_text = self.encoder.to_radicals(source)

            data.append({
                'translation': {
                    'zh': source_text,
                    'en': row['english']
                }
            })

        return Dataset.from_list(data)

    def fine_tune(self, train_df, val_df, representation, output_dir,
                  num_epochs=5, batch_size=8, learning_rate=5e-5):
        """Fine-tune model on specific representation"""

        print(f"\n{'='*80}")
        print(f"Fine-tuning: {representation} | {len(train_df)} examples | {num_epochs} epochs")
        print(f"{'='*80}\n")

        # Load model
        tokenizer = MarianTokenizer.from_pretrained(self.base_model)
        model = MarianMTModel.from_pretrained(self.base_model)

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df, representation)
        val_dataset = self.prepare_dataset(val_df, representation)

        def preprocess(examples):
            inputs = [ex['zh'] for ex in examples['translation']]
            targets = [ex['en'] for ex in examples['translation']]

            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
            labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=['translation'])
        val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=['translation'])

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_total_limit=2,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            report_to='none',
            logging_steps=10,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train
        trainer.train()

        # Save
        final_dir = f"{output_dir}/final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        print(f"\n✓ Fine-tuning complete: {final_dir}\n")

        return model, tokenizer, trainer.state.log_history

class MultiRepresentationTranslator:
    """Translate using different representations"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = LinguisticEncoder()

    def translate_batch(self, texts: List[str], representation='baseline', max_length=512):
        """Translate a batch of texts"""

        if representation == 'baseline':
            source_texts = texts
        elif representation == 'pinyin':
            source_texts = [self.encoder.to_pinyin(t) for t in texts]
        elif representation == 'pinyin_no_tone':
            source_texts = [self.encoder.to_pinyin(t, with_tone=False) for t in texts]
        elif representation == 'morphemes':
            source_texts = [self.encoder.segment_morphemes(t) for t in texts]
        elif representation == 'radicals':
            source_texts = [self.encoder.to_radicals(t) for t in texts]

        inputs = self.tokenizer(source_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        translated = self.model.generate(**inputs, max_length=max_length)
        translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return translations

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(representations=['baseline', 'morphemes', 'pinyin'],
                   output_dir='./models', save_freq=1):
    """Run the complete low-resource MT experiment"""

    print(f"\n{'#'*80}")
    print(f"# STARTING EXPERIMENT")
    print(f"# Representations: {representations}")
    print(f"# Train sizes: {list(splits.keys())}")
    print(f"{'#'*80}\n")

    all_results = []
    finetuner = RepresentationFineTuner()

    for split_name, split_data in splits.items():
        train_size = split_name.split('_')[1]

        for rep in representations:
            print(f"\n{'*'*80}")
            print(f"* {rep} @ {train_size} examples")
            print(f"{'*'*80}")

            model_output_dir = f"{output_dir}/{rep}_{train_size}"

            # Fine-tune
            model, tokenizer, history = finetuner.fine_tune(
                split_data['train'],
                split_data['val'],
                representation=rep,
                output_dir=model_output_dir,
                num_epochs=5,
                batch_size=4,
                learning_rate=5e-5
            )

            # Move to GPU and evaluate
            model = model.cuda()
            translator = MultiRepresentationTranslator(model, tokenizer)

            predictions = translator.translate_batch(
                split_data['test']['chinese'].tolist(),
                representation=rep
            )

            scores = evaluate_translations(
                predictions,
                split_data['test']['english'].tolist(),
                metric='both'
            )

            result = {
                'representation': rep,
                'train_size': int(train_size),
                'bleu': scores['bleu_corpus'],
                'bleu_std': scores['bleu_sentence_std'],
                'bertscore_f1': scores['bertscore_f1'],
                'bertscore_precision': scores['bertscore_precision'],
                'bertscore_recall': scores['bertscore_recall'],
                'model_path': f"{model_output_dir}/final"
            }

            all_results.append(result)

            print(f"\n{'='*80}")
            print(f"RESULTS: {rep} @ {train_size}")
            print(f"  BLEU: {scores['bleu_corpus']:.2f}")
            print(f"  BERTScore F1: {scores['bertscore_f1']:.4f}")
            print(f"{'='*80}\n")

            # Save intermediate results
            if len(all_results) % save_freq == 0:
                pd.DataFrame(all_results).to_csv('finetuned_results_intermediate.csv', index=False)

            # Free memory
            del model, tokenizer, translator
            torch.cuda.empty_cache()

    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('finetuned_results_FINAL.csv', index=False)

    print(f"\n{'#'*80}")
    print("# EXPERIMENT COMPLETE")
    print(f"{'#'*80}\n")
    print(results_df.to_string(index=False))

    return results_df

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    REPRESENTATIONS = ['baseline', 'morphemes', 'pinyin']
    OUTPUT_DIR = './models_supercomputer'

    # Run experiment
    results = run_experiment(
        representations=REPRESENTATIONS,
        output_dir=OUTPUT_DIR,
        save_freq=1
    )

    print("\n✓ Results saved to: finetuned_results_FINAL.csv")
    print(f"✓ Models saved to: {OUTPUT_DIR}/")
    print("\nDone!")
