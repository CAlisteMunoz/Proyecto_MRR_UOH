#!/bin/bash
#SBATCH --job-name=MRR_PROD
#SBATCH --output=logs/run_%j.log
#SBATCH --error=logs/error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=ngen-ko
#SBATCH --account=caliste

source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate mrr_env

cd $SLURM_SUBMIT_DIR
python src/main.py
