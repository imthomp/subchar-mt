# Fine-Tuning Experiment – BYU Supercomputer Submission

## Project Structure

```
hanzi-decomposition/
├── encoder.py                  # Shared LinguisticEncoder (used by both scripts)
├── main.py                     # Local demo: runs all encoding strategies
├── finetune_experiment.py      # Fine-tuning experiment (runs on supercomputer)
├── run_finetuning.sh           # SLURM batch script
├── data/
│   └── ids.txt                 # CHISE Ideographic Description Sequences
├── requirements.txt            # Light deps for local use
└── requirements-finetune.txt   # Full deps for supercomputer
```

## Quick Start

### 1. Edit the SLURM script

Open `run_finetuning.sh` and update your email:
```bash
#SBATCH --mail-user=YOUR_NETID@byu.edu
```

### 2. Transfer files to the supercomputer

```bash
# From your local machine
scp -r encoder.py finetune_experiment.py run_finetuning.sh data/ YOUR_NETID@ssh.fsl.byu.edu:~/cs501r/
```

### 3. SSH into the supercomputer

```bash
ssh YOUR_NETID@ssh.fsl.byu.edu
cd ~/cs501r/
```

### 4. Set up Python environment (first time only)

```bash
module load python/3.10
module load cuda/11.8

python -m venv ~/venv_mt_finetune
source ~/venv_mt_finetune/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-finetune.txt
```

### 5. Submit the job

```bash
sbatch run_finetuning.sh
```

### 6. Monitor the job

```bash
squeue -u $USER                          # Check job status
tail -f finetune_*.out                   # Watch output in real-time
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"  # GPU info
```

### 7. Retrieve results

```bash
# From your local machine
scp YOUR_NETID@ssh.fsl.byu.edu:~/cs501r/finetuned_results_FINAL.csv .
scp YOUR_NETID@ssh.fsl.byu.edu:~/cs501r/logs/experiment_*.log .
```

## What the Experiment Does

1. Loads WMT19 Chinese-English dataset (2,000 examples)
2. Creates low-resource splits: 100 / 500 / 1,000 training examples
3. Fine-tunes `Helsinki-NLP/opus-mt-zh-en` on each representation × data size:
   - **baseline** – raw Chinese characters
   - **morphemes** – jieba word segmentation
   - **pinyin** – phonological romanisation
4. Evaluates with BLEU and BERTScore
5. Saves results to `finetuned_results_FINAL.csv`

## Configuration

Edit `finetune_experiment.py` to change experiment parameters:

```python
# Which representations to test
REPRESENTATIONS = ['baseline', 'morphemes', 'pinyin']  # add 'radicals' etc.

# Training data sizes
splits, test_data, val_data = create_low_resource_splits(
    df_full,
    train_sizes=[100, 500, 1000]  # e.g. add 50
)

# Epochs / batch size (inside fine_tune() call in run_experiment)
num_epochs=5,
batch_size=4,
```

## Expected Runtime

| GPU   | Estimated time |
|-------|---------------|
| A100  | ~45 min – 1 hr |
| V100  | ~1.5 – 2 hrs |

Total models: 9 (3 representations × 3 data sizes)

## Output Files

| File | Description |
|------|-------------|
| `finetuned_results_FINAL.csv` | All results (BLEU, BERTScore) |
| `finetuned_results_intermediate.csv` | Checkpoint saves |
| `finetune_*.out` | SLURM stdout log |
| `finetune_*.err` | SLURM stderr log |
| `logs/experiment_*.log` | Full experiment log |
| `models_supercomputer/*/` | Trained model checkpoints |

## Troubleshooting

**Job fails immediately** – check `finetune_*.err`; verify modules loaded and venv is activated.

**Out of memory** – reduce `batch_size` (try `2` instead of `4`).

**Job times out** – increase `#SBATCH --time` in `run_finetuning.sh` (e.g. `06:00:00`).

**Dataset download fails** – WMT19 requires internet access; contact BYU RC if needed.

## For Class Submission

Submit:
1. `Lab_Exploration_1_FOR_CLASS.ipynb` – pre-trained experiments
2. `finetune_experiment.py` + `encoder.py` – fine-tuning code
3. `run_finetuning.sh` – SLURM script
4. `finetuned_results_FINAL.csv` – supercomputer results
5. `logs/experiment_*.log` – evidence it ran
6. PDF report: what you did, what you learned, ~5 hours of work
