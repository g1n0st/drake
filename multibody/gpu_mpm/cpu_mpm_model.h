#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/cuda_mpm_model.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

// NOTE(changyu): `CpuMpmModel` is responsive to store the initial config in `DeformableModel`,
// (mesh topology, particle state, material/solver parameters, etc.),
// and use them to initialize the GpuMpmState when all finalize.
// TODO (changyu): now it only be specific for MPM cloth.
template<typename T>
struct CpuMpmModel {
    CpuMpmModel() = default;
    std::vector<Vec3<T>> pos;
    std::vector<Vec3<T>> vel;
    std::vector<int> indices;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake