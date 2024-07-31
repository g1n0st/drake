#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/cuda_mpm_model.cuh"
#include "multibody/gpu_mpm/cuda_mpm_kernels.cuh"
#include "multibody/gpu_mpm/radix_sort.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

template<typename T>
void GpuMpmState<T>::InitializeQRCloth(const std::vector<Vec3<T>> &pos,
                                         const std::vector<Vec3<T>> &vel,
                                         const std::vector<int> &indices) {
    n_verts_ = pos.size();
    n_faces_ = indices.size() / 3;
    assert(n_faces_ * 3  == indices.size());
    n_particles_ = n_verts_ + n_faces_;

    h_positions_ = pos;
    h_velocities_ = vel;

    h_indices_ = indices;
    // NOTE (changyu): at the initial state, position/velocity is organized as [n_faces | n_verts].
    for (auto &v : h_indices_) {
        v += n_faces_;
    }

    // device particle buffer allocation for reorder data
    for (uint32_t i = 0; i < 2; ++i) {
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_positions, sizeof(Vec3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_velocities, sizeof(Vec3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_volumes, sizeof(T) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_affine_matrices, sizeof(Mat3<T>) * n_particles_));

        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_pids, sizeof(int) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_sort_keys, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_sort_ids, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_sort_keys, 0, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_sort_ids, 0, sizeof(uint32_t) * n_particles_));

        if (i == current_particle_buffer_id_) {
            std::vector<int> id_sequence(n_particles_);
            std::iota(id_sequence.begin(), id_sequence.end(), 0);
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_pids, 
                                      id_sequence.data(), 
                                      sizeof(int) * n_particles_, 
                                      cudaMemcpyHostToDevice));

            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_positions + n_faces_ * 3, 
                                      h_positions_.data(), 
                                      sizeof(Vec3<T>) * n_verts_, 
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_velocities + n_faces_ * 3, 
                                      h_velocities_.data(), 
                                      sizeof(Vec3<T>) * n_verts_, 
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_volumes, 0, sizeof(T) * n_particles_));
            CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_affine_matrices, 0, sizeof(Mat3<T>) * n_particles_));
        }
    }
    
    // scratch data
    CUDA_SAFE_CALL(cudaMalloc(&d_forces_, sizeof(Vec3<T>) * n_particles_));
    CUDA_SAFE_CALL(cudaMalloc(&d_taus_, sizeof(Mat3<T>) * n_particles_));
    CUDA_SAFE_CALL(cudaMalloc(&d_index_mappings_, sizeof(int) * n_particles_));

    // element-based data
    CUDA_SAFE_CALL(cudaMalloc(&d_deformation_gradients_, sizeof(Mat3<T>) * n_faces_));
    CUDA_SAFE_CALL(cudaMalloc(&d_Dm_inverses_, sizeof(Mat2<T>) * n_faces_));
    CUDA_SAFE_CALL(cudaMalloc(&d_indices_, sizeof(int) * n_faces_ * 3));
    CUDA_SAFE_CALL(cudaMemcpy(d_indices_, h_indices_.data(), sizeof(int) * n_faces_ * 3, cudaMemcpyHostToDevice));


    // device grid buffer allocation
    // NOTE(changyu): considering the problem size, we pre-allocate the dense grid once and skip the untouched parts when traversal.
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_masses, config::G_DOMAIN_VOLUME * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_momentum, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_touched_flags, config::G_GRID_VOLUME * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_touched_ids, config::G_GRID_VOLUME * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_touched_cnt, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(grid_buffer_.d_g_touched_cnt, 0, sizeof(uint32_t)));

    radix_sort(this->next_sort_keys(), this->current_sort_keys(), this->next_sort_ids(), this->current_sort_ids(), sort_buffer_, sort_buffer_size_, static_cast<unsigned int>(n_particles_));
    CUDA_SAFE_CALL(cudaMalloc(&sort_buffer_, sizeof(unsigned int) * sort_buffer_size_));

    CUDA_SAFE_CALL((
        initialize_fem_state_kernel<<<
        (this->n_faces() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (this->n_faces(), this->indices(), this->current_positions(), this->current_velocities(), this->current_volumes(),
         this->deformation_gradients(), this->Dm_inverses())
        ));
}

template<typename T>
void GpuMpmState<T>::Destroy() {
    for (uint32_t i = 0; i < 2; ++i) {
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_positions));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_velocities));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_volumes));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_affine_matrices));

        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_pids));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_sort_keys));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_sort_ids));

        // make sure to throw error when illegal access happens
        particle_buffer_[i].d_positions = nullptr;
        particle_buffer_[i].d_velocities = nullptr;
        particle_buffer_[i].d_volumes = nullptr;
        particle_buffer_[i].d_affine_matrices = nullptr;
        particle_buffer_[i].d_pids = nullptr;
        particle_buffer_[i].d_sort_keys = nullptr;
        particle_buffer_[i].d_sort_ids = nullptr;
    }

    CUDA_SAFE_CALL(cudaFree(d_forces_));
    CUDA_SAFE_CALL(cudaFree(d_taus_));
    CUDA_SAFE_CALL(cudaFree(d_index_mappings_));
    CUDA_SAFE_CALL(cudaFree(d_deformation_gradients_));
    CUDA_SAFE_CALL(cudaFree(d_Dm_inverses_));
    CUDA_SAFE_CALL(cudaFree(d_indices_));
    d_forces_ = nullptr;
    d_taus_ = nullptr;
    d_index_mappings_ = nullptr;
    d_deformation_gradients_ = nullptr;
    d_Dm_inverses_ = nullptr;
    d_indices_ = nullptr;

    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_masses));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_momentum));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_touched_flags));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_touched_ids));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_touched_cnt));
    grid_buffer_.d_g_masses = nullptr;
    grid_buffer_.d_g_momentum = nullptr;
    grid_buffer_.d_g_touched_flags = nullptr;
    grid_buffer_.d_g_touched_ids = nullptr;
    grid_buffer_.d_g_touched_cnt = nullptr;

    CUDA_SAFE_CALL(cudaFree(sort_buffer_));
    sort_buffer_ = nullptr;
    sort_buffer_size_ = 0;
}

template class GpuMpmState<double>;
template class GpuMpmState<float>;

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake