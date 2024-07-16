#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/cuda_mpm_model.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

// NOTE(changyu): this solver should be stateless, all the required data should be initialized and stored in `GpuMpmState`.
// NOTE(changyu): `GpuMpmSolver` is responsive to launch cuda kernels in `cuda_mpm_kernels.cuh`.

template<typename T>
class GpuMpmSolver {
public:
    void RebuildMapping(GpuMpmState<T> *state) {
        state = state;
    }
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake