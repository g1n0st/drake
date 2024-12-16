#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <chrono>
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
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_rigid_id(), contact_pairs.non_mpm_id.data(), sizeof(uint32_t) * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_pos(), contact_pairs.particle_in_contact_position.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_dist(), contact_pairs.penetration_distance.data(), sizeof(T) * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_normal(), contact_pairs.normal.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_rigid_v(), contact_pairs.rigid_v.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(state->contact_rigid_p_WB(), contact_pairs.rigid_p_WB.data(), sizeof(T) * 3 * n_contacts, cudaMemcpyHostToDevice));
    this->GpuSync();
    CUDA_SAFE_CALL((
        initialize_contact_velocities<T, config::DEFAULT_CUDA_BLOCK_SIZE><<<
        (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (n_contacts, state->contact_vel(), state->contact_mpm_id(), state->current_velocities())
        ));
    this->GpuSync();
}

template<typename T>
void GpuMpmSolver<T>::UpdateContact(GpuMpmState<T> *state, const int frame, const T& dt, const T& friction_mu, const T& stiffness, const T& damping) const {
    const auto &n_contacts = state->num_contacts();
    if (!n_contacts) return;

    const uint32_t &touched_blocks_cnt = state->grid_touched_cnt_host();
    const uint32_t &touched_cells_cnt = touched_blocks_cnt * config::G_BLOCK_VOLUME;

    // statistics
    std::vector<T> s_residuals;
    std::vector<T> s_times;
    std::vector<T> s_energies;

    CUDA_SAFE_CALL((
        compute_base_cell_node_index_kernel<<<
        (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (n_contacts, state->contact_pos(), state->contact_sort_keys(), state->contact_sort_ids())
        ));
    
    const bool dump = true;
    const int max_newton_iterations = 200;
    constexpr bool use_jacobi = true;
    const T kTol = 1e-5;

    bool enable_line_search = true;
    const T jacobi_relax_coeff = 1.0;
    const bool global_line_search = use_jacobi;
    const bool exact_line_search = true;
    int count = 0;

    T norm_dir = 1e10;
    T *norm_dir_d = nullptr;
    T global_E0 = T(0.);
    T *global_E0_d = nullptr;
    T global_E1 = T(0.);
    T *global_E1_d = nullptr;
    T *global_dE1_d = nullptr;
    T *global_d2E1_d = nullptr;
    int grid_DoFs = 0;
    int line_search_cnt = 0;
    uint32_t total_grid_DoFs = 0;
    uint32_t *total_grid_DoFs_d = nullptr;
    uint32_t solved_grid_DoFs = 0;
    uint32_t *solved_grid_DoFs_d = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&norm_dir_d, sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&total_grid_DoFs_d, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&solved_grid_DoFs_d, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&global_E0_d, sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&global_E1_d, sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&global_dE1_d, sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&global_d2E1_d, sizeof(T)));

    // NOTE (changyu): pre-compute contact particle velocity `contact_vel0` after p2g2g before contact handling
    // then the dv changed by the implicit contact optimization problem can be extacted by `dv = contact_vel - contact_vel`.
    CUDA_SAFE_CALL((
        grid_to_particle_kernel<T, config::DEFAULT_CUDA_BLOCK_SIZE, /*CONTACT_TRANSFER=*/true><<<
        (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (n_contacts, state->contact_pos(), state->contact_vel0(), nullptr,
        state->grid_masses(), state->grid_momentum(), dt)
        ));

    while (norm_dir > kTol && count < max_newton_iterations) {
        long long before_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        CUDA_SAFE_CALL(cudaMemset(norm_dir_d, 0, sizeof(T)));
        CUDA_SAFE_CALL(cudaMemset(global_E0_d, 0, sizeof(T)));
        CUDA_SAFE_CALL(cudaMemset(global_E1_d, 0, sizeof(T)));
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
        for (uint32_t color_mask = 0U; color_mask < (use_jacobi ? 1U: 27U); ++color_mask) {
            CUDA_SAFE_CALL((
                contact_particle_to_grid_kernel<T, 32, use_jacobi><<<
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
                update_grid_contact_coordinate_descent_kernel<T, use_jacobi><<<
                (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                state->grid_v_star(), state->grid_Hess(), state->grid_Grad(), state->grid_momentum(), state->grid_Dir(),
                state->grid_alpha(), state->grid_E0(), state->grid_E1(),
                norm_dir_d, total_grid_DoFs_d, color_mask, jacobi_relax_coeff)
                ));
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            CUDA_SAFE_CALL(cudaMemcpy(&total_grid_DoFs, total_grid_DoFs_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            // printf("color(%u) total_grid_DoFs=%u\n", color_mask, total_grid_DoFs);
            
            // line search
            const auto & line_search = [&](const T current_alpha) {
                CUDA_SAFE_CALL(cudaMemset(global_E1_d, 0, sizeof(T)));
                CUDA_SAFE_CALL(cudaMemset(global_dE1_d, 0, sizeof(T)));
                CUDA_SAFE_CALL(cudaMemset(global_d2E1_d, 0, sizeof(T)));
                CUDA_SAFE_CALL((
                    grid_to_particle_vdb_line_search_kernel<T, 32, /*JACOBI=*/true, /*SOLVE_DF_DDF=*/true><<<
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
                    nullptr,
                    global_E1_d,
                    global_dE1_d,
                    global_d2E1_d,
                    dt, friction_mu, stiffness, damping, color_mask,
                    /*eval_E0=*/ false,
                    global_line_search,
                    current_alpha)
                    ));
                CUDA_SAFE_CALL((
                    update_global_energy_grid_kernel<T, true, /*SOLVE_DF_DDF=*/true><<<
                    (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                    (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                    state->grid_v_star(), state->grid_momentum(), state->grid_Dir(),
                    nullptr, global_E1_d, global_dE1_d, global_d2E1_d,
                    /*color_mask*/ 0, current_alpha, /*eval_E0=*/ false)
                    ));
                CUDA_SAFE_CALL(cudaDeviceSynchronize());
                T E, dE, d2E;
                CUDA_SAFE_CALL(cudaMemcpy(&E, global_E1_d, sizeof(T), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(&dE, global_dE1_d, sizeof(T), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(&d2E, global_d2E1_d, sizeof(T), cudaMemcpyDeviceToHost));
                return std::make_tuple(E, dE, d2E);
            };

            const auto sign = [](T value) -> int {
                if (value > 0) return 1;
                if (value < 0) return -1;
                return 0;
            };

            line_search_cnt = 0;

            // NOTE (changyu): DoNewtonWithBisectionFallback reference:
            // https://github.com/RobotLocomotion/drake/blob/1e19d4808c684f28cadd86e3f5d4387e571de75f/multibody/contact_solvers/newton_with_bisection.cc
            // https://github.com/RobotLocomotion/drake/blob/5fa497cfde7496910eed84ba54755bf2e555284f/multibody/contact_solvers/sap/sap_solver.cc#L682-L686
            // This relative tolerance was obtained by experimentation on a large set of
            // tests cases. We found out that with f_tolerance ∈ [10⁻¹⁴, 10⁻³] the solver
            // is robust with small changes in performances (about 30%). We then choose a
            // safe tolerance far enough from the lower limit (close to machine epsilon)
            // and the upper limit (close to an inexact method).
            const T f_tolerance = T(1e-8);
            // NOTE (changyu): for exact line search, take jacobi_relax_coeff as alpha_guess
            const T x_tolerance = f_tolerance * jacobi_relax_coeff; // alpha_tolerance = f_tolerance * alpha_guess;
            T global_alpha = T(1.);
            T x_lower = T(0.), x_upper = T(1.), root;
            std::tuple<T, T, T> f_lower, f_upper, f_root;
            if (enable_line_search && global_line_search && exact_line_search) {
                f_lower = line_search(T(0.));
                f_upper = line_search(T(1.));
            }
            if (std::get<1>(f_lower) < T(0.) && std::get<1>(f_upper) < T(0.)) { // pick alpha=1.0 is optimal when grad<0 is always true
                x_lower = T(1.0);
                f_lower = f_upper;
            }
            root = x_lower; // Initialize to user supplied guess.
            T minus_dx = x_lower - x_upper;
            T minus_dx_previous = minus_dx;

            // Helper to perform a bisection update. It returns the pair (root, -dx).
            auto do_bisection = [&x_lower, &x_upper]() {
                const T dx_negative = T(.5) * (x_lower - x_upper);
                // N.B. This way of updating the root will lead to root == x_lower if
                // the value of minus_dx is insignificant compared to x_lower when using
                // floating point precision.
                const T x = x_lower - dx_negative;
                return std::make_pair(x, dx_negative);
            };

            // Helper to perform a Newton update. It returns the pair (root, -dx).
            auto do_newton = [&f_root, &root]() {
                const T dx_negative = std::get<1>(f_root) / std::get<2>(f_root);
                T x = root;
                // N.B. x will not change if dx_negative is negligible within machine
                // precision.
                x -= dx_negative;
                return std::make_pair(x, dx_negative);
            };

            bool global_line_search_satisfied = false;
            while (!(((enable_line_search && global_line_search) && global_line_search_satisfied) || 
                     ((!global_line_search || !enable_line_search) && solved_grid_DoFs == total_grid_DoFs))) {
                if (enable_line_search && global_line_search && exact_line_search) {
                    // The one evaluation per iteration.
                    f_root = line_search(root);

                    // Update the bracket around root to guarantee that there exist a root
                    // within the interval [x_lower, x_upper].
                    // Bisection
                    if (sign(std::get<1>(f_root)) != sign(std::get<1>(f_upper))) {
                        x_lower = root;
                        f_lower = f_root;
                    } else {
                        x_upper = root;
                        f_upper = f_root;
                    }

                    // Exit if f(root) is close to zero.
                    if (abs(std::get<1>(f_root)) < f_tolerance) {
                        global_line_search_satisfied = true;
                    }

                    // N.B. This check is based on the check used within method rtsafe from
                    // Numerical Recipes.
                    // N.B. One way to think about this: if we assume 0 ≈ |fᵏ| << |fᵏ⁻¹| and
                    // f'ᵏ⁻¹ ≈ f'ᵏ (this would be the case when Newton is converging
                    // quadratically), then we can estimate fᵏ⁻¹ from values at the last
                    // iteration as fᵏ⁻¹ ≈ fᵏ + dxᵏ⁻¹⋅f'ᵏ⁻¹ ≈ dxᵏ⁻¹⋅f'ᵏ. Therefore the
                    // inequality below is an approximation for |2⋅fᵏ| > |fᵏ⁻¹|. That is, we use
                    // Newton's method when |fᵏ| < |fᵏ⁻¹|/2. Otherwise we use bisection which
                    // guarantees convergence, though linearly.
                    const bool newton_is_slow = T(2.) * abs(std::get<1>(f_root)) > abs(minus_dx_previous * std::get<2>(f_root));

                    minus_dx_previous = minus_dx;
                    if (newton_is_slow) {
                        std::tie(root, minus_dx) = do_bisection();
                    }
                    else {
                        std::tie(root, minus_dx) = do_newton();
                        if (x_lower <= root && root <= x_upper) {
                        } else {
                             std::tie(root, minus_dx) = do_bisection();
                        }
                    }

                    // No need for additional evaluations if the root is within tolerance.
                    if (abs(minus_dx) < x_tolerance) {
                        global_line_search_satisfied = true;
                    }
                }
                else if (enable_line_search) {
                    CUDA_SAFE_CALL(cudaMemset(global_E1_d, 0, sizeof(T)));
                    CUDA_SAFE_CALL((
                        grid_to_particle_vdb_line_search_kernel<T, 32, use_jacobi, /*SOLVE_DF_DDF=*/false><<<
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
                        global_line_search ? global_E0_d : state->grid_E0(),
                        global_line_search ? global_E1_d : state->grid_E1(),
                        nullptr, nullptr,
                        dt, friction_mu, stiffness, damping, color_mask,
                        /*eval_E0=*/line_search_cnt == 0,
                        global_line_search,
                        global_alpha)
                        ));
                }

                if (enable_line_search && global_line_search) {
                    if (exact_line_search && global_line_search_satisfied) {
                        global_E1 = std::get<0>(f_root);
                        global_alpha = root;
                    } else {
                        CUDA_SAFE_CALL((
                            update_global_energy_grid_kernel<T, use_jacobi, /*SOLVE_DF_DDF=*/false><<<
                            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                            state->grid_v_star(), state->grid_momentum(), state->grid_Dir(),
                            global_E0_d, global_E1_d, nullptr, nullptr,
                            color_mask, global_alpha, /*eval_E0=*/line_search_cnt == 0)
                            ));
                        CUDA_SAFE_CALL(cudaDeviceSynchronize());
                        CUDA_SAFE_CALL(cudaMemcpy(&global_E0, global_E0_d, sizeof(T), cudaMemcpyDeviceToHost));
                        CUDA_SAFE_CALL(cudaMemcpy(&global_E1, global_E1_d, sizeof(T), cudaMemcpyDeviceToHost));
                        if (global_E1 <= global_E0) { // NOTE (changyu): numerical error on float
                            global_line_search_satisfied = true;
                            // printf("global line search E0=%.7f E1=%.7f alpha=%.3f\n", global_E0, global_E1, global_alpha);
                        } else {
                            global_alpha /= T(2.0);
                            if (global_alpha < T(1e-8)) {
                                printf("Tiny Alpha!!!!!!!!!!! E0=%.10f E1=%.10f\n", global_E0, global_E1);
                                global_line_search_satisfied = true;
                            }
                        }
                    }
                    if (global_line_search_satisfied) {
                        s_energies.push_back(global_E1);
                        CUDA_SAFE_CALL((
                            apply_global_line_search_grid_kernel<T, use_jacobi><<<
                            (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                            (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                            state->grid_momentum(), state->grid_Dir(),
                            color_mask, global_alpha)
                            ));
                    }
                } else {
                    CUDA_SAFE_CALL((
                        update_grid_contact_alpha_kernel<T, use_jacobi><<<
                        (touched_cells_cnt + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
                        (touched_cells_cnt, state->grid_touched_ids(), state->grid_masses(),
                        state->grid_v_star(), state->grid_momentum(), state->grid_Dir(),
                        state->grid_alpha(), state->grid_E0(), state->grid_E1(),
                        solved_grid_DoFs_d, color_mask, enable_line_search)
                        ));
                    CUDA_SAFE_CALL(cudaDeviceSynchronize());
                    CUDA_SAFE_CALL(cudaMemcpy(&solved_grid_DoFs, solved_grid_DoFs_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                }

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
        long long after_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        s_residuals.push_back(norm_dir);
        s_times.push_back(T((after_ts-before_ts) / 1e3));
    }
    // throw;
    std::cout << "Iteration count :" <<  count 
              << ", tol: " << norm_dir 
              << ", n_contacts " << n_contacts 
              << ", grid_DoFs " << grid_DoFs 
              << ", line_search_cnt " << line_search_cnt
              << std::endl;
    CUDA_SAFE_CALL(cudaFree(norm_dir_d));
    CUDA_SAFE_CALL(cudaFree(total_grid_DoFs_d));
    CUDA_SAFE_CALL(cudaFree(solved_grid_DoFs_d));

    if (dump) {
        std::ofstream file("/home/changyu/Desktop/mpm-data/" 
                           + std::string(use_jacobi ? "jacobi" : "colored_gs") 
                           + "_iter_" + std::to_string(max_newton_iterations)
                           + "_frame_" + std::to_string(frame) 
                           + ".json");
        file << "[\n";
        for (int i = 0; i < count; ++i) {
            file << "  {\n";
            file << "      \"time\": " << s_times[i] << ",\n";
            if (enable_line_search && global_line_search) {
                file << "      \"residual\": " << s_residuals[i] << ",\n";
                file << "      \"energy\": " << s_energies[i] << "\n";
            } else {
                file << "      \"residual\": " << s_residuals[i] << "\n";
            }
            file << "  }";
            if (i != count -1) file << ",";
            file << "\n";
        }
        file << "]\n";
        file.close();
        printf("Dumped\n");
    }

    // NOTE (changyu): two-way coupling part, apply contact impulse back to the rigid part
    CUDA_SAFE_CALL((apply_contact_impulse_to_rigid_bodies<T><<<
        (n_contacts + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (n_contacts, state->contact_pos(), state->contact_vel0(), state->contact_vel(), 
        state->current_volumes(), state->contact_mpm_id(), state->contact_rigid_id(), 
        state->contact_rigid_p_WB(), state->F_Bq_W_tau(), state->F_Bq_W_f())
        ));
}

template class GpuMpmSolver<config::GpuT>;

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake