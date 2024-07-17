#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/cuda_mpm_solver.cuh"
#include "multibody/gpu_mpm/cuda_mpm_kernels.cuh"
#include "multibody/gpu_mpm/radix_sort.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

template<typename T>
void GpuMpmSolver<T>::RebuildMapping(GpuMpmState<T> *state) {
    compute_base_cell_node_index<<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), state->current_positions(), state->current_sort_keys(), state->current_sort_ids());
    radix_sort(state->next_sort_keys(), state->current_sort_keys(), state->next_sort_ids(), state->current_sort_ids(), state->sort_buffer(), state->sort_buffer_size(), static_cast<unsigned int>(state->n_particles()));
    compute_sorted_state<<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), 
         state->current_positions(), state->current_velocities(), state->current_masses(), state->current_deformation_gradients(), state->current_affine_matrices(),
         state->next_sort_ids(),
         state->next_positions(),
         state->next_velocities(), state->next_masses(), state->next_deformation_gradients(), state->next_affine_matrices());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    state->SwitchCurrentState();
}

template class GpuMpmSolver<double>;
template class GpuMpmSolver<float>;

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake