#!/bin/bash
#SBATCH --job-name=MRR_EXP_W
#SBATCH --output=logs/run_exp_w_%j.log
#SBATCH --error=logs/error_exp_w_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=ngen-ko
#SBATCH --account=caliste

source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate mrr_env

cd $SLURM_SUBMIT_DIR

echo "Comenzando batería de experimentos de núcleos de convolución para W..."
python src/experimento_w.py
echo "Procesamiento finalizado."
