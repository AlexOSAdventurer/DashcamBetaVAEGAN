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

## Change the arguments in the SLURM and python files
For the SLURM files:
1. Change the singularity container path to whatever you need.
2. Change the output log path to one that also works for you.
3. Select the correct GPU partition for your SLURM system, and make sure you do a proper resource provision for you.
3. For different SLURM files you will have to change various different environment variables to change effectively arguments being passed into the container. Further details below for each file.

For the python files:
1. Change any necessary numerical constants (think batch size or learning rate) that might affect the behavior of the script. Further details below.

# Training the beta-VAE GAN
We use the ```train_torchgan.slurm``` file for this task.
There are two major variables/variable groups in the SLURM script to change as needed:
1. VAE\_KL\_WEIGHT - this is the beta variable.
2. LOAD\_PREFIX and SAVE\_PREFIX - you can use this to restart a training job and save a training job respectively. As you can see I usually have these correspond to the job names.

In the ```gpu_train_pure_torchgan.py``` which is the script that the slurm script invokes, there are other variables at the top of the file to pay attention to:
1. ```batch_size``` and ```learning_rate```, which are what you may normally expect.
2. ```num_epochs``` - your max epoch limit
3. ```zDim``` latent space dimensionality, and also a proportionality constant within the models themselves - smaller the number, the smaller your models will be.

All of these in the python script are already set to what I use, so to replicate my results you shouldn't have to change anything.

# Getting the latent space analysis that I did
Invoke the ```single_gpu_latent.bash``` file while being in a well-provisioned node with at least one GPU. 
1. Create the ```latent_analysis``` folder in your current working directory.
2. Change the ```LOAD\_PREFIX``` variable within the script to change the model that is laoded and executed.
3. Run the ```single_gpu_latent.bash``` script.

This will create an list of images like in the paper - reach row represents one latent variable being changed up and down based off a seed image (central column is the original image unchanged), with the left-most column with the latent variable much less than it was originally (-50 decrease), and the right-most column having that latent variable significantly increased (+50 increase).
To make the generation of images easier, the images are split across several files, with 32 latent variables being tweaked in each file - you can reference the "index" annotation of each file, and the index number of the original image as well, which are both annotated in the filename of each image file.
