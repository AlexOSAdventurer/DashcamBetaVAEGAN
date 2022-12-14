#!/bin/bash

export LOAD_PREFIX=CommaAiFinalApproach-zdim512-4worlds-beta100-2e5

singularity exec --nv $WORK/torchgan_container_hpc_old_latest.sif python3 ./display_dataset_image.py
