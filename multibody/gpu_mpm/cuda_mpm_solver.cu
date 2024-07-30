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
void GpuMpmSolver<T>::RebuildMapping(GpuMpmState<T> *state, bool sort) const {
    // TODO, NOTE (changyu):
    // Since we currently adopt dense grid, it's exactly as extending Section 4.2.1 Rebuild-Mapping in [Fei et.al 2021]:
    // "One can push to use more neighboring blocks than we do, and the extreme would end up with a dense background grid,
    // where the rebuild mapping can be removed entirely."
    // NOTE (changyu): Otherwise, this RebuildMapping could somehow be the bottleneck:
    // "In our experiments (Fig. 6), when the number of particles is small, i.e., 55.3k, the rebuild-mapping
    // itself is the bottleneck, and our free zone scheme alone brings 3.7× acceleration."
    CUDA_SAFE_CALL((
        compute_base_cell_node_index_kernel<<<
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
    if (sort) {
        // NOTE, TODO (changyu): radix sort with the first 16 bits is good enough to balance the performance between P2G and sort itself,
        // but more tuning could be conducted here.
        CUDA_SAFE_CALL((
            radix_sort(state->next_sort_keys(), 
                       state->current_sort_keys(), 
                       state->next_sort_ids(), 
                       state->current_sort_ids(), 
                       state->sort_buffer(), 
                       state->sort_buffer_size(), 
                       static_cast<unsigned int>(state->n_particles()),
                       /*num_bit = */ std::min((config::G_DOMAIN_BITS * 3), 16))
            ));
        CUDA_SAFE_CALL((
            compute_sorted_state_kernel<<<
            (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
            (state->n_particles(), 
            state->current_positions(), state->current_velocities(), state->current_volumes(), state->current_affine_matrices(),
            state->next_sort_ids(),
            state->next_positions(),
            state->next_velocities(), state->next_volumes(), state->next_affine_matrices())
            ));
        state->SwitchCurrentState();
    }
}

template<typename T>
void GpuMpmSolver<T>::CalcFemStateAndForce(GpuMpmState<T> *state, const T& dt) const {
    CUDA_SAFE_CALL(cudaMemset(state->forces(), 0, sizeof(Vec3<T>) * state->n_particles()));
    CUDA_SAFE_CALL(cudaMemset(state->taus(), 0, sizeof(Mat3<T>) * state->n_particles()));

    CUDA_SAFE_CALL((
        calc_fem_state_and_force_kernel<<<
        (state->n_faces() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_faces(), state->indices(), state->current_volumes(), state->current_affine_matrices(),
         state->current_positions(), state->current_velocities(), state->deformation_gradients(),
         state->forces(), state->taus(), dt)
        ));
}

template<typename T>
void GpuMpmSolver<T>::ParticleToGrid(GpuMpmState<T> *state, const T& dt) const {
    const uint32_t &touched_blocks_cnt = state->grid_touched_cnt_host();
    const uint32_t &touched_cells_cnt = touched_blocks_cnt * config::G_BLOCK_VOLUME;
    if (touched_cells_cnt > 0) {
    CUDA_SAFE_CALL((
        clean_grid_kernel<<<
        (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (touched_cells_cnt, state->grid_touched_ids(), state->grid_touched_flags(), state->grid_masses(), state->grid_momentum())
        ));
    }
    CUDA_SAFE_CALL((
        particle_to_grid_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), state->current_positions(), state->current_velocities(), state->current_volumes(), state->current_affine_matrices(),
         state->forces(), state->taus(),
         state->current_sort_keys(),
         state->grid_touched_flags(), state->grid_masses(), state->grid_momentum(), dt)
        ));
}

template<typename T>
void GpuMpmSolver<T>::UpdateGrid(GpuMpmState<T> *state) const {
    // NOTE (changyu): we gather the grid block that are really touched
    CUDA_SAFE_CALL(cudaMemset(state->grid_touched_cnt(), 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL((
        gather_touched_grid_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (config::G_GRID_VOLUME + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->grid_touched_flags(), state->grid_touched_ids(), state->grid_touched_cnt(), state->grid_masses())
        ));

    const uint32_t &touched_blocks_cnt = state->grid_touched_cnt_host();
    const uint32_t &touched_cells_cnt = touched_blocks_cnt * config::G_BLOCK_VOLUME;
    CUDA_SAFE_CALL((
        update_grid_kernel<<<
        (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(), state->grid_momentum())
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