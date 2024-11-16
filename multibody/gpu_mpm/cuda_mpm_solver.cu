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
    // NOTE (changyu):
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

    // NOTE (changyu):
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
        // NOTE (changyu): radix sort with the first 16 bits is good enough to balance the performance between P2G and sort itself
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
            state->current_positions(), state->current_velocities(), state->current_volumes(), state->current_affine_matrices(), state->current_pids(),
            state->next_sort_ids(),
            state->next_positions(),
            state->next_velocities(), state->next_volumes(), state->next_affine_matrices(), state->next_pids(), state->index_mappings())
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
        (state->n_faces(), state->indices(), state->index_mappings(), state->current_volumes(), state->current_affine_matrices(), state->Dm_inverses(),
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
void GpuMpmSolver<T>::UpdateGrid(GpuMpmState<T> *state, int mpm_bc) const {
    // NOTE (changyu): we gather the grid block that are really touched
    CUDA_SAFE_CALL(cudaMemset(state->grid_touched_cnt(), 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL((
        gather_touched_grid_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (config::G_GRID_VOLUME + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->grid_touched_flags(), state->grid_touched_ids(), state->grid_touched_cnt(), state->grid_masses())
        ));

    const uint32_t &touched_blocks_cnt = state->grid_touched_cnt_host();
    const uint32_t &touched_cells_cnt = touched_blocks_cnt * config::G_BLOCK_VOLUME;

    if (mpm_bc == 0) {
        CUDA_SAFE_CALL((
            update_grid_kernel<T, 0><<<
            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(), state->grid_momentum(), state->grid_v_star())
            ));
    } else if (mpm_bc == 1) {
        CUDA_SAFE_CALL((
            update_grid_kernel<T, 1><<<
            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(), state->grid_momentum(), state->grid_v_star())
            ));
    } else if (mpm_bc == 2) {
        CUDA_SAFE_CALL((
            update_grid_kernel<T, 2><<<
            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(), state->grid_momentum(), state->grid_v_star())
            ));
    } else if (mpm_bc == 3) {
        CUDA_SAFE_CALL((
            update_grid_kernel<T, 3><<<
            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(), state->grid_momentum(), state->grid_v_star())
            ));
    } else {
        CUDA_SAFE_CALL((
            update_grid_kernel<T, -1><<<
            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(), state->grid_momentum(), state->grid_v_star())
            ));
    }
}

template<typename T>
void GpuMpmSolver<T>::GridToParticle(GpuMpmState<T> *state, const T& dt) const {
    CUDA_SAFE_CALL((
        grid_to_particle_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE, /*CONTACT_TRANSFER=*/false><<<
        (state->n_particles() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (state->n_particles(), state->current_positions(), state->current_velocities(), state->current_affine_matrices(),
         state->grid_masses(), state->grid_momentum(), dt)
        ));
}

template<typename T>
void GpuMpmSolver<T>::GpuSync() const {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename T>
void GpuMpmSolver<T>::Dump(const GpuMpmState<T> &state, std::string filename) const {
    const auto &dumped_state = state.DumpCpuState();

    std::ofstream obj(filename);
    for (size_t i = 0; i < state.n_verts(); ++i) {
      const auto &vert = std::get<0>(dumped_state)[i];
      obj << "v " << vert[0] << " " << vert[1] << " " << vert[2] << "\n";
    }
    for (size_t i = 0; i < state.n_faces(); ++i) {
      obj << "f " << std::get<1>(dumped_state)[i*3+0]+1 
          << " "  << std::get<1>(dumped_state)[i*3+1]+1 
          << " "  << std::get<1>(dumped_state)[i*3+2]+1 << "\n";
    }
    obj.close();
}

// NOTE (changyu): this method is used to synchroize mpm position states to CPU and get `MpmParticleContactPair`
template<typename T>
void GpuMpmSolver<T>::SyncParticleStateToCpu(GpuMpmState<T> *state) const {
    this->GpuSync();
    state->positions_host().resize(state->n_particles());
    CUDA_SAFE_CALL(cudaMemcpy(state->positions_host().data(), state->current_positions(), sizeof(Vec3<T>) * state->n_particles(), cudaMemcpyDeviceToHost));
}

template<typename T>
void GpuMpmSolver<T>::CopyContactPairs(GpuMpmState<T> *state, const MpmParticleContactPairs<T> &contact_pairs) const {
    const size_t n_contacts = contact_pairs.non_mpm_id.size();
    state->ReallocateContacts(n_contacts);
    if (n_contacts == 0) return;
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_mpm_id(), contact_pairs.particle_in_contact_index.data(), sizeof(uint32_t) * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_pos(), contact_pairs.particle_in_contact_position.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_dist(), contact_pairs.penetration_distance.data(), sizeof(T) * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_normal(), contact_pairs.normal.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_rigid_v(), contact_pairs.rigid_v.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    this->GpuSync();
    CUDA_SAFE_CALL((
        initialize_contact_velocities<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (n_contacts, state->contact_vel(), state->contact_mpm_id(), state->current_velocities())
        ));
    this->GpuSync();
}

template<typename T>
void GpuMpmSolver<T>::UpdateContact(GpuMpmState<T> *state, const T& dt, const T& friction_mu, const T& stiffness, const T& damping) const {
    const auto &n_contacts = state->num_contacts();
    if (!n_contacts) return;

    const uint32_t &touched_blocks_cnt = state->grid_touched_cnt_host();
    const uint32_t &touched_cells_cnt = touched_blocks_cnt * config::G_BLOCK_VOLUME;

    CUDA_SAFE_CALL((
        compute_base_cell_node_index_kernel<<<
        (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (n_contacts, state->contact_pos(), state->contact_sort_keys(), state->contact_sort_ids())
        ));
    
    bool enable_line_search = false;
    const int max_newton_iterations = 100;
    const T kTol = 1e-5;
    int count = 0;
    T norm_dir = 1e10;
    T *norm_dir_d;
    int grid_DoFs = 0;
    uint32_t total_grid_DoFs = 0;
    uint32_t *total_grid_DoFs_d = 0;
    uint32_t solved_grid_DoFs = 0;
    uint32_t *solved_grid_DoFs_d = 0;
    CUDA_SAFE_CALL(cudaMalloc(&norm_dir_d, sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&total_grid_DoFs_d, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&solved_grid_DoFs_d, sizeof(uint32_t)));
    while (norm_dir > kTol && count < max_newton_iterations) {
        CUDA_SAFE_CALL(cudaMemset(norm_dir_d, 0, sizeof(T)));
        grid_DoFs = 0;
        if (touched_cells_cnt > 0) {
            CUDA_SAFE_CALL((
                clean_grid_contact_kernel<<<
                (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                (touched_cells_cnt, state->grid_touched_ids(), 
                state->grid_Hess(), state->grid_Grad(), state->grid_Dir(), 
                state->grid_alpha(), state->grid_E0(), state->grid_E1())
                ));
        }
        for (uint32_t color_mask = 0U; color_mask < 27U; ++color_mask) {
            CUDA_SAFE_CALL((
                contact_particle_to_grid_kernel<T, 32><<<
                (n_contacts + 32 - 1) / 32, 32>>>
                (n_contacts, 
                state->contact_pos(), 
                state->contact_vel(), 
                state->current_velocities(),
                state->current_volumes(),
                state->contact_mpm_id(), 
                state->contact_dist(), 
                state->contact_normal(), 
                state->contact_rigid_v(),
                state->contact_sort_keys(), 
                state->grid_Hess(),
                state->grid_Grad(),
                dt, friction_mu, stiffness, damping, color_mask)
                ));
            CUDA_SAFE_CALL(cudaMemset(total_grid_DoFs_d, 0, sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMemset(solved_grid_DoFs_d, 0, sizeof(uint32_t)));
            solved_grid_DoFs = 0;
            CUDA_SAFE_CALL((
                update_grid_contact_coordinate_descent_kernel<<<
                (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                state->grid_v_star(), state->grid_Hess(), state->grid_Grad(), state->grid_momentum(), state->grid_Dir(),
                state->grid_alpha(), state->grid_E0(), state->grid_E1(),
                norm_dir_d, total_grid_DoFs_d, color_mask)
                ));
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            CUDA_SAFE_CALL(cudaMemcpy(&total_grid_DoFs, total_grid_DoFs_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            // printf("color(%u) total_grid_DoFs=%u\n", color_mask, total_grid_DoFs);
            
            // line search
            int line_search_cnt = 0;
            while (solved_grid_DoFs < total_grid_DoFs) {
                if (enable_line_search) {
                    if (line_search_cnt == 0) {
                        CUDA_SAFE_CALL((
                            grid_to_particle_vdb_line_search_kernel<T, 32, /*EVAL_E0=*/true><<<
                            (n_contacts + 32 - 1) / 32, 32>>>
                            (n_contacts, 
                            state->contact_pos(), 
                            state->contact_vel(), 
                            state->current_velocities(),
                            state->current_volumes(),
                            state->contact_mpm_id(), 
                            state->contact_dist(), 
                            state->contact_normal(), 
                            state->contact_rigid_v(),
                            state->grid_momentum(),
                            state->grid_Dir(),
                            state->grid_alpha(),
                            state->grid_E0(),
                            state->grid_E1(),
                            dt, friction_mu, stiffness, damping, color_mask)
                            ));
                    } else {
                        CUDA_SAFE_CALL((
                            grid_to_particle_vdb_line_search_kernel<T, 32, /*EVAL_E0=*/false><<<
                            (n_contacts + 32 - 1) / 32, 32>>>
                            (n_contacts, 
                            state->contact_pos(), 
                            state->contact_vel(), 
                            state->current_velocities(),
                            state->current_volumes(),
                            state->contact_mpm_id(), 
                            state->contact_dist(), 
                            state->contact_normal(), 
                            state->contact_rigid_v(),
                            state->grid_momentum(),
                            state->grid_Dir(),
                            state->grid_alpha(),
                            state->grid_E0(),
                            state->grid_E1(),
                            dt, friction_mu, stiffness, damping, color_mask)
                            ));
                    }
                }
                CUDA_SAFE_CALL((
                    update_grid_contact_alpha_kernel<<<
                    (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                    (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                    state->grid_v_star(), state->grid_momentum(), state->grid_Dir(),
                    state->grid_alpha(), state->grid_E0(), state->grid_E1(),
                    solved_grid_DoFs_d, color_mask, enable_line_search)
                    ));
                CUDA_SAFE_CALL(cudaDeviceSynchronize());
                CUDA_SAFE_CALL(cudaMemcpy(&solved_grid_DoFs, solved_grid_DoFs_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                line_search_cnt += 1;
            }
            
            CUDA_SAFE_CALL((
                grid_to_particle_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE, /*CONTACT_TRANSFER=*/true><<<
                (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                (n_contacts, state->contact_pos(), state->contact_vel(), nullptr,
                state->grid_masses(), state->grid_momentum(), dt)
                ));
            // throw;
            grid_DoFs += total_grid_DoFs;
        }

        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaMemcpy(&norm_dir, norm_dir_d, sizeof(T), cudaMemcpyDeviceToHost));
        norm_dir = sqrt(norm_dir) / grid_DoFs;
        count += 1;
        // throw;
    }
    // throw;
    std::cout << "Iteration count :" <<  count 
              << ", tol: " << norm_dir 
              << ", n_contacts " << n_contacts 
              << ", grid_DoFs " << grid_DoFs << std::endl;
    CUDA_SAFE_CALL(cudaFree(norm_dir_d));
    CUDA_SAFE_CALL(cudaFree(total_grid_DoFs_d));
    CUDA_SAFE_CALL(cudaFree(solved_grid_DoFs_d));
}

template class GpuMpmSolver<config::GpuT>;

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake