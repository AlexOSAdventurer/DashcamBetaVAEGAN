# Introduction
Hi Dr. Berger! This is all the source code I've used for running all the experiments and analyses for this class project fo rtesting beta-VAE GANs for generating/encoding dashcam images.
Lots of this code is fairly messy, I know, but I've done a lot of experimentation, so this is (with some scrubbing of comments) verbatim what I used to generate the results in the paper.
# Intended platform
This was initially run on the CRANE cluster of the Holland Computing Center at the University of Nebraska-Lincoln, on their Tesla V100 nodes using an x86-64 platform.
The HCC is largely a standard supercomputer platform that uses SLURM to execute jobs, and uses Singularity as their containerization solution.
Thus, the code here should have their SLURM scripts appropriately adjusted for whatever new supercomputer system you intend to run them on, and the target platform should have singularity available and be an x86-64 system.
# Preliminary Steps

## Platform to be on
Use a SLURM compatible x86-64 supercomputer platform with Singularity available (basically fancy Docker) and NVIDIA GPUs. This code with the default settings has been targeted for the V100s with 32 GB of video memory.
## Download the needed singularity container
I provide a standard singularity container that has the pre-existing userspace necessary for all the code in this project to work correctly -
    singularity pull docker://catvehiclecdt2021/torchgan_container_hpc_old
This will create the torchgan_container_hpc_old_latest.sif file in your current working directory.
## Download the needed data
I use the Berkeley Deep Drive dataset, with 70k images for training, and 10k images for validation. I reshape the images into 128x128 RGB images that are compiled into large numpy array files which are directly used for training.
There are two ways to do this:
1. Download the bdd100k images directly from the available source, and run the ```compile_files_into_numpy.py``` and then the ```turn_smaller.py``` python scripts (while on a well-provisioned node) to create the training and validation files yourself. I won't describe how to do this because I've already done this for you and strongly suggest step 2 instead.
2. Downlaod the compiled npy files I've already created directly from box. This file is already provided for you. You can directly unzip it into your project folder and use it as I use it. Warning: The resulting files are about 30 GB, since it is uncompressed images. Box link:
