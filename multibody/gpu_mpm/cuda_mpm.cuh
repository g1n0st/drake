#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>

#include <eigen3/Eigen/Dense>

#include "multibody/gpu_mpm/settings.h"

namespace drake {
namespace multibody {
namespace gmpm {

template <typename T>
struct GpuMpmState {

    GpuMpmState() = default;

    // NOTE (changyu): initialize GPU MPM state, all gpu memory allocation should be done here to avoid re-allocation.
    void InitializeParticles(const std::vector<Vec3<T>> &pos, const std::vector<Vec3<T>> &vel, const T& mass) {
        n_particles = pos.size();

        h_positions = pos;
        h_velocities = vel;
        h_masses.resize(n_particles, mass);
        h_deformation_gradients.resize(n_particles, Mat3<T>::Identity());
        h_affine_matrices.resize(n_particles, Mat3<T>::Zero());

        // device particle buffer allocation
        for (uint32_t i = 0; i < 2; ++i) {
            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_positions, sizeof(Vec3<T>) * n_particles));
            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_velocities, sizeof(Vec3<T>) * n_particles));
            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_masses, sizeof(T) * n_particles));
            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_deformation_gradients, sizeof(Mat3<T>) * n_particles));
            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_affine_matrices, sizeof(Mat3<T>) * n_particles));

            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_sort_keys, sizeof(uint64_t) * n_particles));
            CUDA_SAFE_CALL(cudaMalloc(&particle_buffer[i].d_sort_ids, sizeof(uint32_t) * n_particles));
            CUDA_SAFE_CALL(cudaMemset(particle_buffer[i].d_sort_keys, 0, sizeof(uint64_t) * n_particles));
            CUDA_SAFE_CALL(cudaMemset(particle_buffer[i].d_sort_ids, 0, sizeof(uint32_t) * n_particles));

            if (i == current_particle_buffer_id) {
                CUDA_SAFE_CALL(cudaMemcpy(particle_buffer[i].d_positions, h_positions.data(), sizeof(Vec3<T>) * n_particles, cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(particle_buffer[i].d_velocities, h_velocities.data(), sizeof(Vec3<T>) * n_particles, cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(particle_buffer[i].d_masses, h_masses.data(), sizeof(T) * n_particles, cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(particle_buffer[i].d_deformation_gradients, h_deformation_gradients.data(), sizeof(Mat3<T>) * n_particles, cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(particle_buffer[i].d_affine_matrices, h_affine_matrices.data(), sizeof(Mat3<T>) * n_particles, cudaMemcpyHostToDevice));
            }
        }

        // device grid buffer allocation
        // NOTE(changyu): considering the problem size, we pre-allocate the dense grid once and skip the untouched parts when traversal.
        CUDA_SAFE_CALL(cudaMalloc(&grid_buffer.d_g_masses, config::G_DOMAIN_VOLUME * sizeof(T)));
        CUDA_SAFE_CALL(cudaMalloc(&grid_buffer.d_g_momentum, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
        CUDA_SAFE_CALL(cudaMalloc(&grid_buffer.d_g_flags, config::G_GRID_VOLUME * sizeof(Vec3<T>)));
    }

    void Destroy() {
        for (uint32_t i = 0; i < 2; ++i) {
            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_positions));
            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_velocities));
            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_masses));
            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_deformation_gradients));
            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_affine_matrices));

            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_sort_keys));
            CUDA_SAFE_CALL(cudaFree(particle_buffer[i].d_sort_ids));

            // make sure to throw error when illegal access happens
            particle_buffer[i].d_positions = nullptr;
            particle_buffer[i].d_velocities = nullptr;
            particle_buffer[i].d_masses = nullptr;
            particle_buffer[i].d_deformation_gradients = nullptr;
            particle_buffer[i].d_affine_matrices = nullptr;
            particle_buffer[i].d_sort_keys = nullptr;
            particle_buffer[i].d_sort_ids = nullptr;
        }

        CUDA_SAFE_CALL(cudaFree(grid_buffer.d_g_masses));
        CUDA_SAFE_CALL(cudaFree(grid_buffer.d_g_momentum));
        CUDA_SAFE_CALL(cudaFree(grid_buffer.d_g_flags));
        grid_buffer.d_g_masses = nullptr;
        grid_buffer.d_g_momentum = nullptr;
        grid_buffer.d_g_flags = nullptr;
    }

    // Particles state device ptrs
    size_t n_particles;
    
    struct ParticleBuffer {
        T* d_positions = nullptr;
        T* d_velocities = nullptr;
        T* d_masses = nullptr;
        T* d_deformation_gradients = nullptr;
        T* d_affine_matrices = nullptr;

        uint64_t* d_sort_keys = nullptr;
        uint32_t* d_sort_ids = nullptr;
    };
    
    uint32_t current_particle_buffer_id = 0;
    std::array<ParticleBuffer, 2> particle_buffer;

    // Particles state host ptrs
    // TODO(changyu): Host memory should be managed by Drake context instead of here.
    std::vector<Vec3<T>> h_positions;
    std::vector<Vec3<T>> h_velocities;
    std::vector<T> h_masses;
    std::vector<Mat3<T>> h_deformation_gradients;
    std::vector<Mat3<T>> h_affine_matrices;

    // Grid state device ptrs

    struct GridBuffer {
        T* d_g_masses;
        T* d_g_momentum;
        uint32_t* d_g_flags;
    };

    GridBuffer grid_buffer;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake