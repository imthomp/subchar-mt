#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=gloss
#SBATCH --output=logs/gloss_%j.out
#SBATCH --error=logs/gloss_%j.err

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "Gloss injection experiment started: $(date)"
echo "Node: $SLURM_NODELIST"

module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p results/gloss_injection logs

PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/gloss_injection.py 2>&1 | tee logs/gloss_injection.log

echo "Done: $(date)"
