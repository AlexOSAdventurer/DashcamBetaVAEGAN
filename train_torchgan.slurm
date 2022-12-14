#!/bin/bash
#SBATCH --job-name=CommaAiResidual27
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --nodes=4
#SBATCH --mem=180gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu_32gb'
#SBATCH --output=/work/cseos2g/papapalpi/DeepDriveStuff/bdd100k/images/100k/%x-%j.out
#SBATCH --wait-all-nodes=1

# SLURM Inherits your environment. cd $SLURM_SUBMIT_DIR not needed
pwd; hostname; date

export MASTER_PORT=12364
export WORLD_SIZE=8
export LOAD_PREFIX=
export SAVE_PREFIX=CommaAiResidual27
export VAE_KL_WEIGHT=1

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun singularity exec --nv $WORK/torchgan_container_hpc_old_latest.sif python3 ./gpu_train_pure_torchgan.py
