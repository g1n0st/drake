# Drake Implementation of A Convex Formulation of Material Points and Rigid Bodies with GPU-Accelerated Async-Coupling for Interactive Simulation (IROS 2025)

We present a novel convex formulation that weakly couples the Material Point Method (MPM) with rigid body dynamics through frictional contact, optimized for efficient GPU parallelization. Our approach features an asynchronous time-splitting scheme to integrate MPM and rigid body dynamics under different time step sizes. We develop a globally convergent quasi-Newton solver tailored for massive parallelization, achieving up to 500x speedup over previous convex formulations without sacrificing stability. Our method enables interactive-rate simulations of robotic manipulation tasks with diverse deformable objects including granular materials and cloth, with strong convergence guarantees. We detail key implementation strategies to maximize performance and validate our approach through rigorous experiments, demonstrating superior speed, accuracy, and stability compared to state-of-the-art MPM simulators for robotics.

[Paper Link](https://arxiv.org/abs/2503.05046)

## Installation

``` bash
git clone -b cuda-mpm-weak-coupling-clean git@github.com:g1n0st/drake.git
```

We currently support **Ubuntu 22.04** with **CUDA 12.1**.

Please follow the official Drake from-source build guide for detailed instructions: https://drake.mit.edu/from_source.html

### 1. Check CUDA Compiler (Required)

Ensure that CUDA 12.1 is properly installed. You can verify it with:

```bash
nvcc --version
```

If not installed, refer to NVIDIA’s CUDA installation guide: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-12-1-0-download-archive)

### 2. Install Drake Build Dependencies
Run the following script to install system dependencies:

``` bash
cd drake
./setup/install_prereqs
```

This script will install all required packages and tools needed to build Drake and our extension.

## Demos

Go to the demo folder by:

``` bash
cd examples/multibody/deformable
```

### Rolling an Elastoplastic Dough

``` bash
bazel run roll --config omp --
```

### Cloth and rigid bodies

``` bash
bazel run allegro_bagging --config omp --
```

### Laundry

``` bash
bazel run allegro_bagging_cloth --config omp --
```

###  Folding and Unfolding a T-Shirt

``` bash
bazel run dual_arm_flipping --config omp --
```
