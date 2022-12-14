#!/bin/bash

export MASTER_PORT=12356
export WORLD_SIZE=1

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export LOAD_PREFIX=
export SAVE_PREFIX=CommaAiResidual13
export VAE_KL_WEIGHT=1

singularity exec --nv $WORK/torchgan_container_hpc_old_latest.sif python3 ./gpu_train_pure_torchgan.py #> "$SAVE_PREFIX.out" &
