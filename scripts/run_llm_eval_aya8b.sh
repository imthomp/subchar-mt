#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --account=ekr8
#SBATCH --job-name=llm_aya8b
#SBATCH --output=logs/llm_aya8b_%j.out
#SBATCH --error=logs/llm_aya8b_%j.err

# Aya Expanse 8B: multilingual judge, reads Chinese source directly
# DO NOT submit until CohereForAI/aya-expanse-8b is fully downloaded on login node

PROJECT_DIR="/home/it238/projects/subchar-mt"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "Aya Expanse 8B LLM eval started: $(date)  node: $SLURM_NODELIST"

module purge
module load python/3.12
module load cuda/12.8.1-jesavxf
module load cudnn/8.9.7.29-12-3s4v3zq

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p results/llm_eval logs

PYTHONPATH="$PROJECT_DIR/src" $PYTHON src/llm_eval.py \
    --preds_dir results/predictions_v3/ \
    --model_id CohereForAI/aya-expanse-8b \
    --n_sentences 100 \
    --train_size 500 \
    2>&1 | tee logs/llm_eval_aya8b.log

echo "Done: $(date)"
