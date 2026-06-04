#!/bin/bash

#SBATCH --time=12:00:00               # v2 has 9 reps × 2 models — give extra time
#SBATCH --array=0-4                   # 5 seeds: task 0→42, 1→123, 2→456, 3→789, 4→999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=mt_finetune_v2
#SBATCH --output=logs/finetune_v2_%A_%a.out
#SBATCH --error=logs/finetune_v2_%A_%a.err
#SBATCH --mail-type=NONE

# ============================================================================
# BYU Supercomputer - Fine-tuning Experiment v2 (Job Array)
# Adds: byte, random_index, sentencepiece representations; 5 seeds; prediction saving
#
# Pre-requisites (run on login node before submitting):
#   1. Train SentencePiece model:
#        source .venv/bin/activate
#        python src/train_sentencepiece.py
#   2. Verify HF cache (models + WMT19 dataset):
#        python -c "from datasets import load_dataset; load_dataset('wmt19','zh-en',split='train[:1]')"
#
# Submit:   sbatch scripts/run_finetuning_v2.sh
# Aggregate (after all tasks finish):
#           python src/aggregate_results.py
# Significance tests:
#           python src/significance_test.py --all_csv finetuned_results_ALL.csv
# ============================================================================

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "Job started:  $(date)"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task index:   $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURM_NODELIST"
echo "========================================================================"

nvidia-smi

# Load modules
module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

# Set up Python environment
echo ""
echo "Setting up Python environment..."
echo "========================================================================"

VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$VENV_DIR" ] || [ ! -f "$PYTHON" ]; then
    echo "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python python3.12
    uv pip install --python "$PYTHON" \
        "torch>=2.6.0" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 --quiet
    uv pip install --python "$PYTHON" \
        -r "$PROJECT_DIR/requirements.txt" --quiet
    echo "Virtual environment created."
fi

echo "Python: $PYTHON"
echo "Python version: $($PYTHON --version)"
echo ""

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p models_supercomputer logs results/predictions

# ============================================================================
# Verify SentencePiece model exists
# ============================================================================
if [ ! -f "$PROJECT_DIR/data/zh_sp.model" ]; then
    echo "ERROR: SentencePiece model not found at data/zh_sp.model"
    echo "Run on a login node first: python src/train_sentencepiece.py"
    exit 1
fi

# ============================================================================
# PRE-CACHE CHECK (offline mode on compute nodes)
# ============================================================================
echo "========================================================================"
echo "Verifying HuggingFace cache..."
echo "========================================================================"

export HF_HUB_OFFLINE=1

$PYTHON - <<'CACHE_EOF'
import sys, os
os.environ['HF_HUB_OFFLINE']      = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE']  = '1'

from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM

missing = []
checks = [
    ("WMT19 zh-en",         lambda: load_dataset('wmt19', 'zh-en', split='train[:1]')),
    ("opus-mt-zh-en",       lambda: MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')),
    ("NLLB-600M tokenizer", lambda: AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')),
]

for name, fn in checks:
    print(f"  {name}... ", end='', flush=True)
    try:
        fn(); print("OK")
    except Exception as e:
        print("MISSING"); missing.append(name)

print("  COMET (wmt22-comet-da)... ", end='', flush=True)
try:
    from comet import download_model
    download_model('Unbabel/wmt22-comet-da')
    print("OK")
except Exception as e:
    print(f"not found (COMET will be skipped): {e}")

if missing:
    print(f"\nERROR: Missing: {missing}", file=sys.stderr)
    sys.exit(1)

print("All assets verified.")
CACHE_EOF

if [ $? -ne 0 ]; then exit 1; fi
echo ""

# ============================================================================
# RUN EXPERIMENT
# ============================================================================
echo "========================================================================"
echo "Starting v2 experiment (task $SLURM_ARRAY_TASK_ID)..."
echo "========================================================================"
echo ""

LOG="logs/experiment_v2_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"

PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/finetune_experiment.py \
    --version v2 \
    --save_predictions \
    --predictions_dir results/predictions \
    2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "After all 5 tasks finish, run:"
    echo "  python src/aggregate_results.py"
    echo "  python src/significance_test.py"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ Task $SLURM_ARRAY_TASK_ID failed (exit code $EXIT_CODE)"
    echo "========================================================================"
    exit $EXIT_CODE
fi

echo ""
echo "========================================================================"
echo "Job finished: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "========================================================================"


