#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/cuda_mpm_model.cuh"
#include "multibody/gpu_mpm/cuda_mpm_kernels.cuh"
#include "multibody/gpu_mpm/radix_sort.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

template<typename T>
void GpuMpmState<T>::InitializeParticles(const std::vector<Vec3<T>> &pos, const std::vector<Vec3<T>> &vel) {
    n_particles_ = pos.size();

    h_positions_ = pos;
    h_velocities_ = vel;
    h_volumes_.resize(n_particles_, config::P_VOLUME);
    h_deformation_gradients_.resize(n_particles_, Mat3<T>::Identity());
    h_affine_matrices_.resize(n_particles_, Mat3<T>::Zero());

    // device particle buffer allocation
    for (uint32_t i = 0; i < 2; ++i) {
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_positions, sizeof(Vec3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_velocities, sizeof(Vec3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_volumes, sizeof(T) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_deformation_gradients, sizeof(Mat3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_affine_matrices, sizeof(Mat3<T>) * n_particles_));

        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_sort_keys, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_sort_ids, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_sort_keys, 0, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_sort_ids, 0, sizeof(uint32_t) * n_particles_));

        if (i == current_particle_buffer_id_) {
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_positions, h_positions_.data(), sizeof(Vec3<T>) * n_particles_, cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_velocities, h_velocities_.data(), sizeof(Vec3<T>) * n_particles_, cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_volumes, h_volumes_.data(), sizeof(T) * n_particles_, cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_deformation_gradients, h_deformation_gradients_.data(), sizeof(Mat3<T>) * n_particles_, cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_affine_matrices, h_affine_matrices_.data(), sizeof(Mat3<T>) * n_particles_, cudaMemcpyHostToDevice));
        }
    }

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
}

template<typename T>
void GpuMpmState<T>::Destroy() {
    for (uint32_t i = 0; i < 2; ++i) {
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_positions));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_velocities));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_volumes));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_deformation_gradients));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_affine_matrices));

        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_sort_keys));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_sort_ids));

        // make sure to throw error when illegal access happens
        particle_buffer_[i].d_positions = nullptr;
        particle_buffer_[i].d_velocities = nullptr;
        particle_buffer_[i].d_volumes = nullptr;
        particle_buffer_[i].d_deformation_gradients = nullptr;
        particle_buffer_[i].d_affine_matrices = nullptr;
        particle_buffer_[i].d_sort_keys = nullptr;
        particle_buffer_[i].d_sort_ids = nullptr;
    }

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