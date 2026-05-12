#!/bin/bash
#SBATCH --job-name=MRR_TEST
#SBATCH --output=logs/test_%j.log
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=ngen-ko
#SBATCH --account=caliste

source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate mrr_env

cd $SLURM_SUBMIT_DIR
python tests/dry_run.py
