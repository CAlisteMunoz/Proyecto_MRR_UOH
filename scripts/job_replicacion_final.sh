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

echo "=== INICIANDO TRABAJO EN EL CLÚSTER ==="
date

# 1. Preparar el entorno
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate mrr_env

cd $SLURM_SUBMIT_DIR

# 2. Ejecutar todas las rutinas de visualización secuencialmente
echo "--------------------------------------------------------"
echo ">> [1/3] Generando Replicación Exacta (Proyecto Base)..."
python src/visualization/replicacion_exacta.py

echo "--------------------------------------------------------"
echo ">> [2/3] Generando Visualizaciones Exclusivas Vf (Paleta Rojiza)..."
python src/visualization/plot_vf_rojizo.py

echo "--------------------------------------------------------"
echo ">> [3/3] Generando Análisis Comparativo de 5 Ventanas..."
python src/visualization/plot_gradientes_5_ventanas.py
echo "--------------------------------------------------------"

echo "=== TRABAJO FINALIZADO CON ÉXITO ==="
date
