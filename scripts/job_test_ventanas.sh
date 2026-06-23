#!/bin/bash
#SBATCH --job-name=MRR_TEST_W
#SBATCH --output=logs/run_test_w_%j.log
#SBATCH --error=logs/error_test_w_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=ngen-ko
#SBATCH --account=caliste

source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate mrr_env

cd $SLURM_SUBMIT_DIR

echo "Ejecutando Test de 5 Archivos x 5 Ventanas para Velocidad de Caída..."
python src/test_ventanas_w.py
echo "¡Test Finalizado Exitosamente!"
