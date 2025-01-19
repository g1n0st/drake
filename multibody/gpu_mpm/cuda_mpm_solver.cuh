#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/cpu_mpm_model.h"
#include "multibody/gpu_mpm/cuda_mpm_model.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

// NOTE(changyu): this solver should be stateless, all the required data should be initialized and stored in `GpuMpmState`.
// NOTE(changyu): `GpuMpmSolver` is responsive to launch cuda kernels in `cuda_mpm_kernels.cuh`.

template<typename T>
class GpuMpmSolver {
public:
    void RebuildMapping(GpuMpmState<T> *state, bool sort) const;
    void CalcFemStateAndForce(GpuMpmState<T> *state, const T& dt) const;
    void ParticleToGrid(GpuMpmState<T> *state, const T& dt) const;
    void UpdateGrid(GpuMpmState<T> *state, int mpm_bc = -1) const;
    void GridToParticle(GpuMpmState<T> *state, const T& dt) const;
    void GpuSync() const;
    void SyncParticleStateToCpu(GpuMpmState<T> *state) const;
    void Dump(const GpuMpmState<T> &state, std::string filename) const;
    void CopyContactPairs(GpuMpmState<T> *state, const MpmParticleContactPairs<T> &contact_pairs) const;
    void UpdateContact(GpuMpmState<T> *state, const int frame, const int substep, const T& dt, const T& friction_mu, const T& stiffness, const T& damping, const bool bump) const;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake