#!/bin/bash

export LOAD_PREFIX=CommaAiFinalApproach-zdim512-4worlds-beta250-2e5

singularity exec --nv $WORK/torchgan_container_hpc_old_latest.sif python3 ./torchgan_analyze_latents.py
