#!/bin/bash
#SBATCH --job-name=gpfit
#SBATCH --output=/users/ariywagh/GP_SN/sn_gp/slurm_logs/gpfit_%j.out
#SBATCH --error=/users/ariywagh/GP_SN/sn_gp/slurm_logs/gpfit_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /users/ariywagh/GP_SN/sn_gp

# activate your env (adjust if you use conda instead of a venv)
source /users/ariywagh/astro_env/bin/activate

# TF: no GPU, don't oversubscribe threads
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 batch_fit.py --bands gri