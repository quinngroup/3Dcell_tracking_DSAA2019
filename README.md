# 🧬 3D Cell Tracking with Dask for IEEE DSAA 2019

This repository hosts the code for our paper accepted in IEEE DSAA 2019: **"Lightweight and Scalable Particle Tracking and Motion Clustering of 3D Cell Trajectories"**. The project provides a robust pipeline for tracking and clustering cell trajectories using various implementations, including serial, local Dask, and Dask on a cluster.

---

## 🌟 Project Overview

This repository features three versions of the code:

- **🔗 Serial Version**: A straightforward implementation for running on a single machine.
- **⚙️ Local Dask Version**: A parallelized version that leverages Dask for local computation.
- **☁️ Dask on Cluster Version**: A scalable implementation for running on distributed computing clusters.

### Key Technologies Used
- **Python 3.6**
- **Scientific Computing Libraries**: NumPy, SciPy, scikit-learn, matplotlib
- **Computer Vision Tools**: OpenCV 3.1 for detection and tracking
- **Parallel and Distributed Computing**:
  - **Dask Arrays & DataFrames** for scalable processing
  - **Dask-ML & Dask-NDArray** for distributed machine learning
  - **Joblib & multiprocessing** for local parallel execution

---

## 📄 Paper Details

The methods and results of this project are detailed in our paper, which can be accessed through the following links:

- **IEEE Download Page**: [Link to IEEE](https://ieeexplore.ieee.org/abstract/document/8964177)
- **arXiv Preprint**: [Link to arXiv](https://arxiv.org/pdf/1908.03775)

### Citation
For those interested in citing this work, please use the following format:

```bibtex
@INPROCEEDINGS{8964177,
  author={Sedigh Fazli, Mojtaba and Stadler, Rachel V. and Alaila, BahaaEddin and Vella, Stephen A. and Moreno, Silvia N. J. and Ward, Gary E. and Quinn, Shannon},
  booktitle={2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={Lightweight and Scalable Particle Tracking and Motion Clustering of 3D Cell Trajectories}, 
  year={2019},
  volume={},
  number={},
  pages={412-421},
  keywords={Large scale 3D Cell Tracking;Motion Trajectories;Geodesic distance;Spectral Clustering;Toxoplasma gondii;Martin Distance},
  doi={10.1109/DSAA.2019.00056}
}
```

---

## 🚀 How to Run the Code

### Step 1: Prepare Your Data
Ensure that all image data is stored locally on your machine. Update the following parameters in the code before execution:

- **`folders`**: Path to the directories containing image slices.
- **`sample_address`**: Path to a sample TIFF image to extract initial parameters.
- **`cluster_numbers`**: The number of clusters to use in the analysis.

Additional parameters can be modified within the script based on your specific application needs.

### Step 2: Execute the Code
After setting up the parameters, run the code and measure the wall time with:
```bash
time python <The_code_filename.py>
```
This command will provide a performance benchmark by displaying the total execution time.

---

## ❓ Why a Single Python File?
We opted to include all functions in a single script to facilitate seamless wall-time computation for the entire pipeline. This structure allows us to efficiently test and compare running times across different implementations.

---

## 🛠️ Prerequisites

Before running the code, ensure you have the following libraries installed:

```bash
pip install numpy scipy scikit-learn matplotlib opencv-python dask dask-ml joblib
```

For the cluster version, additional Dask configurations may be required.

---

## ✨ Customization Tips
- **Cluster Configuration**: Adjust Dask scheduler and client settings to optimize performance on distributed systems.
- **Memory Management**: For large datasets, tweak Dask's memory handling parameters to prevent overflow.

---

## 📧 Contact
For questions, feedback, or collaboration opportunities, feel free to reach out:
- 📧 mfazli@stanford.edu
- 📧 mfazli@meei.harvard.edu

🚀 Dive into the code and explore the powerful capabilities of lightweight and scalable 3D cell tracking!
