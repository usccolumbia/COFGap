#!/bin/bash
#SBATCH --job-name=cofgap_pi
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=angel,diamond,evodev,quantum
#SBATCH --output=/scratch/projects/cofgap/logs/pi_%A_%a.out
B=/scratch/projects/cofgap
export TABPFN_MODEL_CACHE_DIR=$B/cache TABPFN_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$B/env/bin/python $B/perm_importance_worker.py $B/data $B/out $SLURM_ARRAY_TASK_ID 5
