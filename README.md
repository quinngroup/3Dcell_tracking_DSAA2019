# 3Dcell_tracking_DSAA2019
This repository contains the codes for the paper we submitted to IEEE DSAA 2019 :
_"Lightweight and Scalable Particle Tracking and Motion Clustering of 3D Cell Trajectories"_

There are 3 versions of our codes: Serial, The local dask and the dask on cluster version. 

We implemented our pipeline using Python 3.6 and associated scientific computing libraries (NumPy, SciPy, scikitlearn, matplotlib). The core of our detection and tracking algorithm used a combination of tools availablein the OpenCV 3.1 computer vision library. For the distributed version of the core platform, we used *Dask Arrays* and other Dask dataframes. Some machine learning tools in the distributed version, are obtained from Dask-ml and Dask-ndarrays libraries. For local parallelization and multiprocessing, we used joblib backend and multiprocessing library of Python.

# How to run the code?
First you need to store all the data on your local machine, then inside the code there are two
parameters that should be changed:

_folders : The address of the folders containing the image slices should be mention here._

_sample_address : The address for a sample Tiff image to extract some parameters_

_cluster_numbers : for specifying the number of clusters_

Also there are other parameters inside the program that can be changed based on the application.

After setting the parameters inside the code, you can run the code and compute the wall time as follows:

`time python <The_code_filename.py>` 
# Why we put all the functions inside 1 python file?

The reason is, we aimed to compute the walltime for the whole pipeline, thus we put all the components inside only one file and created a long script. Then we computed the running time in different cases.
