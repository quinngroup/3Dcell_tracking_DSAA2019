# 3Dcell_tracking_DSAA2019
This repository contains the codes for the paper we submitted to IEEE DSAA 2019 :
_"Lightweight and Scalable Particle Tracking and Motion Clustering of 3D Cell Trajectories"_

There are 3 version of our codes: Serial, The local dask and the dask on cluster version. 

# How to run the code?
First you need to store all the data on your local machine, then inside the code there are two
parameters that should be changed:

_folders : The address of the folders containing the image slices should be mention here._

_sample_address : The address for a sample Tiff image to extract some parameters_

_cluster_numbers : for specifying the number of clusters_

Also there are other parameters inside the program that can be changed based on the application.

# Why we put all the functions inside 1 python file?

The reason is, we aimed to compute the walltime for the whole pipeline, thus we put all the components inside only one file and created a long script. Then we computed the running time in different cases.
