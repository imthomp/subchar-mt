#!/bin/bash

#SBATCH --time=04:00:00              # Max time (4 hours - plenty of buffer)
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks=1                    # Single task
#SBATCH --cpus-per-task=4             # 4 CPU cores
#SBATCH --mem=32G                     # 32GB RAM
#SBATCH --gres=gpu:1                  # 1 GPU (will likely get A100)
#SBATCH --partition=gpu               # GPU partition
#SBATCH --qos=gpu                     # GPU QoS
#SBATCH --job-name=mt_finetune        # Job name
#SBATCH --output=finetune_%j.out      # Output file (%j = job ID)
#SBATCH --error=finetune_%j.err       # Error file
#SBATCH --mail-type=END,FAIL          # Email on completion/failure
#SBATCH --mail-user=YOUR_EMAIL@byu.edu  # Replace with your email!

# ============================================================================
# BYU Supercomputer - Fine-tuning Experiment
# Linguistically-Informed Low-Resource Machine Translation
# ============================================================================

echo "========================================================================"
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"

# Print GPU info
nvidia-smi

# Load modules (adjust based on BYU supercomputer setup)
module purge
module load python/3.10           # Or whatever Python version is available
module load cuda/11.8             # Or latest CUDA
module load cudnn/8.6             # CuDNN for GPU acceleration

# Alternative if using conda:
# module load miniconda3
# conda activate your_env_name

# Set up Python environment
echo ""
echo "Setting up Python environment..."
echo "========================================================================"

# Create virtual environment (first time only - comment out after first run)
# python -m venv ~/venv_mt_finetune
# source ~/venv_mt_finetune/bin/activate

# Or activate existing environment
source ~/venv_mt_finetune/bin/activate

# Install required packages (first time only - comment out after first run)
# pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers datasets evaluate sacrebleu bert-score
# pip install pypinyin zhon jieba pandas numpy matplotlib seaborn scipy

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directory
mkdir -p models_supercomputer
mkdir -p logs

# Run the experiment
echo "========================================================================"
echo "Starting fine-tuning experiment..."
echo "========================================================================"
echo ""

python finetune_experiment.py 2>&1 | tee logs/experiment_${SLURM_JOB_ID}.log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Experiment completed successfully!"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ Experiment failed with error code $?"
    echo "========================================================================"
    exit 1
fi

# Print results summary
echo ""
echo "========================================================================"
echo "FINAL RESULTS:"
echo "========================================================================"
if [ -f finetuned_results_FINAL.csv ]; then
    cat finetuned_results_FINAL.csv
else
    echo "Warning: finetuned_results_FINAL.csv not found"
fi

echo ""
echo "========================================================================"
echo "Job finished: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "========================================================================"

# Deactivate virtual environment
deactivate
