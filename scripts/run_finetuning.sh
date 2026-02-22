#!/bin/bash

#SBATCH --time=08:00:00               # Per-task time (each task runs 1 seed)
#SBATCH --array=0-2                   # 3 seeds: task 0→seed 42, 1→123, 2→456
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --job-name=mt_finetune
#SBATCH --output=logs/finetune_%A_%a.out   # %A = array job ID, %a = task index
#SBATCH --error=logs/finetune_%A_%a.err
#SBATCH --mail-type=NONE

# ============================================================================
# BYU Supercomputer - Fine-tuning Experiment (Job Array)
# Linguistically-Informed Low-Resource Machine Translation
#
# Submit with:   sbatch scripts/run_finetuning.sh
# Array tasks run in parallel; each handles one seed.
# After all tasks finish, aggregate results with:
#   python src/aggregate_results.py
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python python3.12
    source "$VENV_DIR/bin/activate"
    uv pip install "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
    uv pip install -r "$PROJECT_DIR/requirements.txt" --quiet
    echo "Virtual environment created and packages installed."
else
    source "$VENV_DIR/bin/activate"
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p models_supercomputer
mkdir -p logs

# ============================================================================
# PRE-CACHE HUGGINGFACE ASSETS
# Compute nodes have no internet. This block checks the local HF cache and
# downloads missing assets (requires network — run from a login node first
# if this fails).
# ============================================================================
echo "========================================================================"
echo "Pre-caching HuggingFace assets..."
echo "========================================================================"

# Keep offline mode on — compute nodes have no internet.
# All assets must be pre-cached on a login node before submitting.
# We just verify here; if anything is missing we fail fast.
export HF_HUB_OFFLINE=1

python - <<'CACHE_EOF'
import sys, os

# Enforce fully offline — no network calls, no retry noise
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE']  = '1'
os.environ['HF_DATASETS_OFFLINE']   = '1'

from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM

missing = []

checks = [
    ("WMT19 zh-en",          lambda: load_dataset('wmt19', 'zh-en', split='train[:1]')),
    ("opus-mt-zh-en",        lambda: MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')),
    ("NLLB-600M tokenizer",  lambda: AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')),
]

for name, fn in checks:
    print(f"  {name}... ", end='', flush=True)
    try:
        fn()
        print("OK")
    except Exception as e:
        print("MISSING")
        missing.append(name)

# COMET is optional
print("  COMET (wmt22-comet-da)... ", end='', flush=True)
try:
    from comet import download_model
    download_model('Unbabel/wmt22-comet-da')
    print("OK")
except Exception as e:
    print(f"not found (COMET scores will be skipped): {e}")

if missing:
    print(f"\nERROR: Missing assets: {missing}", file=sys.stderr)
    print("Pre-cache on a login node:", file=sys.stderr)
    print("  source ~/projects/subchar-mt/.venv/bin/activate  # or: uv run ...", file=sys.stderr)
    print("  python -c \"from datasets import load_dataset; load_dataset('wmt19','zh-en',split='train')\"", file=sys.stderr)
    print("  python -c \"from transformers import AutoTokenizer,AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')\"", file=sys.stderr)
    sys.exit(1)

print("All assets verified.")
CACHE_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# ============================================================================
# RUN EXPERIMENT (task index selects the seed)
# ============================================================================
echo "========================================================================"
echo "Starting experiment (array task $SLURM_ARRAY_TASK_ID)..."
echo "========================================================================"
echo ""

LOG="logs/experiment_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
PYTHONPATH="$PROJECT_DIR/src" python src/finetune_experiment.py 2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Task $SLURM_ARRAY_TASK_ID completed successfully!"
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

deactivate
