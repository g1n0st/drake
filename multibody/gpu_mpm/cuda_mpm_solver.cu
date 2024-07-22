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
void GpuMpmSolver<T>::RebuildMapping(GpuMpmState<T> *state) const {
    // TODO, NOTE (changyu):
    // Since we currently adopt dense grid, it's exactly as extending Section 4.2.1 Rebuild-Mapping in [Fei et.al 2021]:
    // "One can push to use more neighboring blocks than we do, and the extreme would end up with a dense background grid,
    // where the rebuild mapping can be removed entirely."
    // NOTE (changyu): Otherwise, this RebuildMapping could somehow be the bottleneck:
    // "In our experiments (Fig. 6), when the number of particles is small, i.e., 55.3k, the rebuild-mapping
    // itself is the bottleneck, and our free zone scheme alone brings 3.7× acceleration."
    CUDA_SAFE_CALL((
        compute_base_cell_node_index<<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), state->current_positions(), state->current_sort_keys(), state->current_sort_ids())
        ));

    // TODO (changyu):
    // as discussed by Gao et al. [2018], a histogram-sort performs more efficiently, 
    // where the keys are computed through concatenating the block index and the cell code.

    // TODO (changyu):
    // The frequency of sorting can be further reduced as in Section 4.2.2 Particle Sorting in [Fei et.al 2021]
    // Furthermore, as the reduction only helps to lessen the atomic operations within each warp, 
    // instead of sorting w.r.t. cells every time step, 
    // we can perform it only when rebuild-mapping happens.
    // Between two rebuild-mappings, we conduct radix sort in each warp before the reduction in P2G transfer
    // ...
    // Our new scheme may present a less optimal particle ordering, 
    // e.g., particles in the same cell can be distributed to several warps,
    // resulting in several atomics instead of one. 
    // However, this performance loss can be compensated well when particle density is not extremely high in each cell.
    CUDA_SAFE_CALL((
        radix_sort(state->next_sort_keys(), state->current_sort_keys(), state->next_sort_ids(), state->current_sort_ids(), state->sort_buffer(), state->sort_buffer_size(), static_cast<unsigned int>(state->n_particles()))
        ));
    CUDA_SAFE_CALL((
        compute_sorted_state<<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), 
         state->current_positions(), state->current_velocities(), state->current_masses(), state->current_deformation_gradients(), state->current_affine_matrices(),
         state->next_sort_ids(),
         state->next_positions(),
         state->next_velocities(), state->next_masses(), state->next_deformation_gradients(), state->next_affine_matrices())
        ));
    state->SwitchCurrentState();
}

template<typename T>
void GpuMpmSolver<T>::ParticleToGrid(GpuMpmState<T> *state, const T& dt) const {
    CUDA_SAFE_CALL(cudaMemset(state->grid_touched_flags(), 0, config::G_GRID_VOLUME * sizeof(uint32_t)));
    // TODO(changyu): we should gather the grid block that are really touched and just clean them, would be done with GridOperaton
    CUDA_SAFE_CALL(cudaMemset(state->grid_masses(), 0, config::G_DOMAIN_VOLUME * sizeof(T)));
    CUDA_SAFE_CALL(cudaMemset(state->grid_momentum(), 0, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL((
        particle_to_grid_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), state->current_positions(), state->current_velocities(), state->current_masses(), state->current_deformation_gradients(), state->current_affine_matrices(), state->current_sort_keys(),
         state->grid_touched_flags(), state->grid_masses(), state->grid_momentum(), dt)
        ));
}

template<typename T>
void GpuMpmSolver<T>::UpdateGrid(GpuMpmState<T> *state) const {
    CUDA_SAFE_CALL((
        update_grid_kernel_naive<<<
        (config::G_DOMAIN_VOLUME + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->grid_touched_flags(), state->grid_masses(), state->grid_momentum())
        ));
}

template<typename T>
void GpuMpmSolver<T>::GridToParticle(GpuMpmState<T> *state, const T& dt) const {
    CUDA_SAFE_CALL((
        grid_to_particle_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), state->current_positions(), state->current_velocities(), state->current_affine_matrices(),
         state->grid_momentum(), dt)
        ));
}

template class GpuMpmSolver<double>;
template class GpuMpmSolver<float>;

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake