#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "multibody/gpu_mpm/settings.h"

namespace drake {
namespace multibody {
namespace gmpm {

__device__ __host__
inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the cube [0, 1024].
__device__ __host__
inline std::uint32_t morton_code(uint3 xyz) noexcept {
    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

template<typename T>
__global__ void compute_base_cell_node_index(const size_t &n_particles, const T* positions, uint32_t* keys, uint32_t* ids) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        T x = positions[idx * 3 + 0];
        T y = positions[idx * 3 + 1];
        T z = positions[idx * 3 + 2];
        uint32_t xi = static_cast<uint32_t>(x * config::DXINV - 0.5);
        uint32_t yi = static_cast<uint32_t>(y * config::DXINV - 0.5);
        uint32_t zi = static_cast<uint32_t>(z * config::DXINV - 0.5);
        uint32_t higher_bit = morton_code({xi >> config::BLOCK_BITS, yi >> config::BLOCK_BITS, zi >> config::BLOCK_BITS});
        uint32_t lower_bit = ((xi & config::G_BLOCK_MASK) << (config::G_BLOCK_BITS * 2)) | ((yi & config::G_BLOCK_MASK) << config::G_BLOCK_BITS) | (zi & config::G_BLOCK_MASK);
        keys[idx] = (higher_bit << (config::G_BLOCK_BITS * 3)) | lower_bit;
        // keys[idx] = (xi << (config::G_DOMAIN_BITS * 2)) | (yi << (config::G_DOMAIN_BITS * 1)) | (zi);
        // printf("%.3lf %.3lf %.3lf %u %u %u high=%u, low=%u, %u\n", x, y, z, xi, yi, zi, higher_bit, lower_bit, keys[idx]);
        ids[idx] = idx;
    }
}

template<typename T>
__global__ void compute_sorted_state(const size_t &n_particles, 
    const T* current_positions, 
    const T* current_velocities,
    const T* current_masses,
    const T* current_deformation_gradients,
    const T* current_affine_matrices,
    const uint32_t* next_sort_ids,
    T* next_positions,
    T* next_velocities,
    T* next_masses,
    T* next_deformation_gradients,
    T* next_affine_matrices
    ) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            next_positions[idx * 3 + i] = current_positions[next_sort_ids[idx] * 3 + i];
            next_velocities[idx * 3 + i] = current_velocities[next_sort_ids[idx] * 3 + i];
        }
        next_masses[idx] = current_masses[next_sort_ids[idx]];

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            next_deformation_gradients[idx * 9 + i] = current_deformation_gradients[next_sort_ids[idx] * 9 + i];
            next_affine_matrices[idx * 9 + i] = current_affine_matrices[next_sort_ids[idx] * 9 + i];
        }
    }
}

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake