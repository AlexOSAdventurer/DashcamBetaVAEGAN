# Introduction
Hi Dr. Berger! This is all the source code I've used for running all the experiments and analyses for this class project.
Lots of this code is fairly messy, I know, but I've done a lot of experimentation, so this is (with some scrubbing of comments) verbatim what I used to generate the results in the paper.
# Intended platform
This was initially run on the CRANE cluster of the Holland Computing Center at the University of Nebraska-Lincoln, on their Tesla V100 nodes using an x86-64 platform.
The HCC is largely a standard supercomputer platform that uses SLURM to execute jobs, and uses Singularity as their containerization solution.
Thus, the code here should have their SLURM scripts appropriately adjusted for whatever new supercomputer system you intend to run them on, and the target platform should have singularity available and be an x86-64 system.
# Preliminary Steps

## Platform to be on