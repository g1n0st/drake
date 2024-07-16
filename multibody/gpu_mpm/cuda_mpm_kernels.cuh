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
        keys[idx] = morton_code({xi, yi, zi});
        ids[idx] = idx;
    }
}

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake