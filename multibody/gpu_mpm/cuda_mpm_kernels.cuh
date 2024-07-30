#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/math_tools.cuh"
namespace drake {
namespace multibody {
namespace gmpm {

template<typename T>
__global__ void initialize_fem_state_kernel(
    const size_t n_faces,
    const int *indices,
    T *positions,
    T *velocities,
    T *volumes,
    T *deformation_gradients,
    T *Dm_inverses) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_faces) {
        int v0 = indices[idx * 3 + 0];
        int v1 = indices[idx * 3 + 1];
        int v2 = indices[idx * 3 + 2];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            positions[idx * 3 + i] = (positions[v0 * 3 + i] + positions[v1 * 3 + i] + positions[v2 * 3 + i]) / 3.;
            velocities[idx * 3 + i] = (velocities[v0 * 3 + i] + velocities[v1 * 3 + i] + velocities[v2 * 3 + i]) / 3.;
        }

        T D0[3], D1[3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            D0[i] = positions[v1 * 3 + i] - positions[v0 * 3 + i];
            D1[i] = positions[v2 * 3 + i] - positions[v0 * 3 + i];
        }
        T Ds[6] {
            D0[0], D1[0],
            D0[1], D1[1],
            D0[2], D1[2]
        };
        T Q[9], R[6];
        givens_QR<3, 2, T>(Ds, Q, R);
        T Dm[4] = {
            R[0], R[1],
            0   , R[3]
        };

        // For co-dimensional mesh, Dm_inverse stores (R(Ds))^{-1} (e.g. isotropic cloth in 3d)
        // The rotational part is discarded.
        inverse2(Dm, &Dm_inverses[idx * 4]);

        T *F = &deformation_gradients[idx * 9];
        F[0] = T(1.); F[1] = T(0.); F[2] = T(0.);
        F[3] = T(0.); F[4] = T(1.); F[5] = T(0.);
        F[6] = T(0.); F[7] = T(0.); F[8] = T(1.);

        T D0xD1[3];
        cross_product3(D0, D1, D0xD1);
        T volume_4 = norm<3>(D0xD1) / T(8.);

        volumes[idx] += volume_4;
        atomicAdd(&volumes[v0], volume_4);
        atomicAdd(&volumes[v1], volume_4);
        atomicAdd(&volumes[v2], volume_4);
    }
}

template<typename T>
__device__ __host__
inline void fixed_corotated_PK1_2D(const T* F, T* dphi_dF) {
    T U[4], sig[4], V[4];
    svd2x2(F, U, sig, V);
    T R[4];
    matmulT<2, 2, 2, T>(U, V, R);
    T J = determinant2(F);
    T Finv[4];
    inverse2(F, Finv);
    dphi_dF[0] = T(2.) * config::MU * (F[0] - R[0]) + config::LAMBDA * (J - T(1.)) * J * Finv[0];
    dphi_dF[1] = T(2.) * config::MU * (F[1] - R[1]) + config::LAMBDA * (J - T(1.)) * J * Finv[2];
    dphi_dF[2] = T(2.) * config::MU * (F[2] - R[2]) + config::LAMBDA * (J - T(1.)) * J * Finv[1];
    dphi_dF[3] = T(2.) * config::MU * (F[3] - R[3]) + config::LAMBDA * (J - T(1.)) * J * Finv[3];
}

template<typename T>
__device__ __host__
inline void compute_dphi_dF(const T* F, T* dphi_dF) {
    // TODO (changyu): optimize local variable usage
    T Q[9], R[9];
    // 0, 1, 2
    // 3, 4, 5
    // 6, 7, 8
    givens_QR<3, 3, T>(F, Q, R);
    T R_hat[4] = {
        R[0], R[1],
        R[3], R[4]
    };
    T dphi_dF_2x2[4];
    fixed_corotated_PK1_2D(R_hat, dphi_dF_2x2);
    T P_hat[9] = {
        dphi_dF_2x2[0], dphi_dF_2x2[1], 0,
        dphi_dF_2x2[2], dphi_dF_2x2[3], 0,
        0             ,              0, 0
    };
    T P_plane[9];
    matmul<3, 3, 3, T>(Q, P_hat, P_plane);

    T rr = R[2] * R[2] + R[5] * R[5];
    T g = config::GAMMA * rr;
    T gp = config::GAMMA;
    T fp = 0;
    if (R[8] < T(1.)) {
        fp = -config::K * (T(1.) - R[8]) * (T(1.) - R[8]);
    }

    T A[9];
    A[0] = gp * R[2] * R[2];
    A[1] = gp * R[2] * R[5];
    A[2] = gp * R[8] * R[2];
    A[4] = gp * R[5] * R[5];
    A[5] = gp * R[8] * R[8];
    A[8] = fp * R[8];
    A[3] = A[1];
    A[6] = A[2];
    A[7] = A[5];

    T P_nonplane[9];
    T Rinv[9], QA[9];
    inverse3(R, Rinv);
    matmul<3, 3, 3, T>(Q, A, QA);
    matmulT<3, 3, 3, T>(QA, Rinv, P_nonplane);
    
    dphi_dF[0] = P_plane[0] + P_nonplane[0];
    dphi_dF[1] = P_plane[1] + P_nonplane[1];
    dphi_dF[2] = P_plane[2] + P_nonplane[2];
    dphi_dF[3] = P_plane[3] + P_nonplane[3];
    dphi_dF[4] = P_plane[4] + P_nonplane[4];
    dphi_dF[5] = P_plane[5] + P_nonplane[5];
    dphi_dF[6] = P_plane[6] + P_nonplane[6];
    dphi_dF[7] = P_plane[7] + P_nonplane[7];
    dphi_dF[8] = P_plane[8] + P_nonplane[8];
}

template<typename T>
__device__ __host__
inline void project_strain(T* F) {
    T Q[9], R[9];
    givens_QR<3, 3, T>(F, Q, R);

    // return mapping
    if (config::GAMMA == T(0.)) { // CASE 1: no friction
        R[8] = min(R[8], T(1.));
        R[2] = T(0.);
        R[5] = T(0.);
    }
    else if (R[8] > T(1.)) {
        R[8] = min(R[8], T(1.));
        R[2] = T(0.);
        R[5] = T(0.);
    }
    else if (R[8] <= T(0.)) { // inversion
        R[2] = T(0.);
        R[5] = T(0.);
        R[8] = max(R[8], T(-1.));
    }
    else {
        T rr = R[2] * R[2] + R[5] * R[5];
        const T gamma_over_k = config::GAMMA / config::K;
        T zz = config::c_F * (R[8] - T(1.)) * (R[8] - T(1.));
        T f = (gamma_over_k * gamma_over_k) * rr - (zz * zz);
        if (f > T(0.)) {
            T c = zz / (gamma_over_k * sqrt(rr));
            R[2] *= c;
            R[5] *= c;
        }
    }

    matmul<3, 3, 3, T>(Q, R, F);
}

template<typename T>
__global__ void calc_fem_state_and_force_kernel(
    const size_t n_faces,
    const int* indices,
    const T* volumes,
    const T* affine_matrices,
    const T* Dm_inverses,
    T* positions, 
    T* velocities,
    T* deformation_gradients,
    T* forces, 
    T* taus,
    const T dt) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_faces) {
        int v0 = indices[idx * 3 + 0];
        int v1 = indices[idx * 3 + 1];
        int v2 = indices[idx * 3 + 2];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            positions[idx * 3 + i] = (positions[v0 * 3 + i] + positions[v1 * 3 + i] + positions[v2 * 3 + i]) / 3.;
            velocities[idx * 3 + i] = (velocities[v0 * 3 + i] + velocities[v1 * 3 + i] + velocities[v2 * 3 + i]) / 3.;
        }

        T* F = &deformation_gradients[idx * 9];
        const T* C = &affine_matrices[idx * 9];
        T ctF[9]; // cotangent F
        ctF[0] = (1.0 + dt * C[0]) * F[0] + dt * C[1] * F[3] + dt * C[2] * F[6];
        ctF[1] = (1.0 + dt * C[0]) * F[1] + dt * C[1] * F[4] + dt * C[2] * F[7];
        ctF[2] = (1.0 + dt * C[0]) * F[2] + dt * C[1] * F[5] + dt * C[2] * F[8];

        ctF[3] = dt * C[3] * F[0] + (1.0 + dt * C[4]) * F[3] + dt * C[5] * F[6];
        ctF[4] = dt * C[3] * F[1] + (1.0 + dt * C[4]) * F[4] + dt * C[5] * F[7];
        ctF[5] = dt * C[3] * F[2] + (1.0 + dt * C[4]) * F[5] + dt * C[5] * F[8];

        ctF[6] = dt * C[6] * F[0] + dt * C[7] * F[3] + (1.0 + dt * C[8]) * F[6];
        ctF[7] = dt * C[6] * F[1] + dt * C[7] * F[4] + (1.0 + dt * C[8]) * F[7];
        ctF[8] = dt * C[6] * F[2] + dt * C[7] * F[5] + (1.0 + dt * C[8]) * F[8];

        project_strain(ctF);

        T d0[3], d1[3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            d0[i] = positions[v1 * 3 + i] - positions[v0 * 3 + i];
            d1[i] = positions[v2 * 3 + i] - positions[v0 * 3 + i];
        }
        T ds[6] {
            d0[0], d1[0],
            d0[1], d1[1],
            d0[2], d1[2]
        };

        T tangent_F[6];
        const T* Dm_inverse = &Dm_inverses[idx * 4];
        matmul<3, 2, 2>(ds, Dm_inverse, tangent_F);
        ctF[0] = tangent_F[0];
        ctF[1] = tangent_F[1];
        ctF[3] = tangent_F[2];
        ctF[4] = tangent_F[3];
        ctF[6] = tangent_F[4];
        ctF[7] = tangent_F[5];

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            F[i] = ctF[i];
        }

        T VP_local[9];
        compute_dphi_dF(ctF, VP_local);
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            VP_local[i] *= volumes[idx];
        }

        // technical document .(15) part 2
        T VP_local_c2[3] = { VP_local[2], VP_local[5], VP_local[8] };
        T ctF_c2[3] = { ctF[2], ctF[5], ctF[8] };
        outer_product<3, T>(VP_local_c2, ctF_c2, &taus[idx * 9]);

        T grad_N_hat[6] = {
            T(-1.), T(1.), T(0.),
            T(-1.), T(0.), T(1.)
        };
        T grad_N[6];
        matmul<2, 2, 3, T>(Dm_inverse, grad_N_hat, grad_N);
        T VP_local_c01[6] = { 
            VP_local[0], VP_local[1],
            VP_local[3], VP_local[4],
            VP_local[6], VP_local[7]
        };

        T G[9];
        matmul<3, 2, 3, T>(VP_local_c01, grad_N, G);

        // technical document .(15) part 1
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            atomicAdd(&forces[v0 * 3 + i], -G[i * 3 + 0]);
            atomicAdd(&forces[v1 * 3 + i], -G[i * 3 + 1]);
            atomicAdd(&forces[v2 * 3 + i], -G[i * 3 + 2]);
        }
    }
}

__device__ __host__
inline std::uint32_t contract_bits(std::uint32_t v) noexcept {
    v &= 0x09249249u;
    v = (v ^ (v >>  2)) & 0x030C30C3u;
    v = (v ^ (v >>  4)) & 0x0300F00Fu;
    v = (v ^ (v >>  8)) & 0xFF0000FFu;
    v = (v ^ (v >> 16)) & 0x000003FFu;
    return v;
}

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
inline std::uint32_t morton_code(const uint3 &xyz) noexcept {
    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

__device__ __host__
inline uint3 inverse_morton_code(const uint32_t &code) noexcept {
    // Implement the inverse of expand_bits to get the original x, y, z values.
    const std::uint32_t x = contract_bits(code >> 2);
    const std::uint32_t y = contract_bits(code >> 1);
    const std::uint32_t z = contract_bits(code);
    return {x, y, z};
}

__device__ __host__
inline std::uint32_t cell_index(const uint32_t &xi, const uint32_t &yi, const uint32_t &zi) noexcept {
    // NOTE (changyu): using morton code ordering within grid block (lower_bit) seems nothing different
    uint32_t higher_bit = morton_code({xi >> config::BLOCK_BITS, yi >> config::BLOCK_BITS, zi >> config::BLOCK_BITS});
    uint32_t lower_bit = ((xi & config::G_BLOCK_MASK) << (config::G_BLOCK_BITS * 2)) | ((yi & config::G_BLOCK_MASK) << config::G_BLOCK_BITS) | (zi & config::G_BLOCK_MASK);
    return (higher_bit << (config::G_BLOCK_BITS * 3)) | lower_bit;
    // printf("%.3lf %.3lf %.3lf %u %u %u high=%u, low=%u, %u\n", x, y, z, xi, yi, zi, higher_bit, lower_bit, keys[idx]);
}

__device__ __host__
inline uint3 inverse_cell_index(const std::uint32_t &index) noexcept {
    // Extract higher_bit and lower_bit
    uint32_t higher_bit = index >> (config::G_BLOCK_BITS * 3);
    uint32_t lower_bit = index & ((1 << (config::G_BLOCK_BITS * 3)) - 1);

    // Extract xi, yi, zi from lower_bit
    uint32_t lower_xi = (lower_bit >> (config::G_BLOCK_BITS * 2)) & config::G_BLOCK_MASK;
    uint32_t lower_yi = (lower_bit >> config::G_BLOCK_BITS) & config::G_BLOCK_MASK;
    uint32_t lower_zi = lower_bit & config::G_BLOCK_MASK;

    // Extract xi, yi, zi from higher_bit using inverse Morton code
    uint3 higher_xyz = inverse_morton_code(higher_bit);

    // Combine higher and lower bits to get original xi, yi, zi
    uint32_t xi = (higher_xyz.x << config::BLOCK_BITS) | lower_xi;
    uint32_t yi = (higher_xyz.y << config::BLOCK_BITS) | lower_yi;
    uint32_t zi = (higher_xyz.z << config::BLOCK_BITS) | lower_zi;

    return {xi, yi, zi};
}

template<typename T>
__global__ void compute_base_cell_node_index_kernel(const size_t n_particles, const T* positions, uint32_t* keys, uint32_t* ids) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        T x = positions[idx * 3 + 0];
        T y = positions[idx * 3 + 1];
        T z = positions[idx * 3 + 2];
        uint32_t xi = static_cast<uint32_t>(x * config::G_DX_INV - 0.5);
        uint32_t yi = static_cast<uint32_t>(y * config::G_DX_INV - 0.5);
        uint32_t zi = static_cast<uint32_t>(z * config::G_DX_INV - 0.5);
        /*uint3 inv_xyz = inverse_cell_index(cell_index(xi, yi, zi));
        if (xi != inv_xyz.x || yi != inv_xyz.y || zi != inv_xyz.z) {
            printf("%u,%u, %u,%u %u,%u\n", xi, inv_xyz.x, yi, inv_xyz.y, zi, inv_xyz.z);
        }*/
        keys[idx] = cell_index(xi, yi, zi);
        ids[idx] = idx;
    }
}

template<typename T>
__global__ void compute_sorted_state_kernel(const size_t n_particles, 
    const T* current_positions, 
    const T* current_velocities,
    const T* current_volumes,
    const T* current_affine_matrices,
    const uint32_t* next_sort_ids,
    T* next_positions,
    T* next_velocities,
    T* next_volumes,
    T* next_affine_matrices
    ) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            next_positions[idx * 3 + i] = current_positions[next_sort_ids[idx] * 3 + i];
            next_velocities[idx * 3 + i] = current_velocities[next_sort_ids[idx] * 3 + i];
        }
        next_volumes[idx] = current_volumes[next_sort_ids[idx]];

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            next_affine_matrices[idx * 9 + i] = current_affine_matrices[next_sort_ids[idx] * 9 + i];
        }
    }
}

template<typename T, int BLOCK_DIM>
__global__ void particle_to_grid_kernel(const size_t n_particles,
    const T* positions, 
    const T* velocities,
    const T* volumes,
    const T* affine_matrices,
    const T* forces, 
    const T* taus,
    const uint32_t* grid_index,
    uint32_t* g_touched_flags,
    T* g_masses,
    T* g_momentum,
    const T dt) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // In [Fei et.al 2021],
    // we spill the B-spline weights (nine floats for each thread) by storing them into the shared memory
    // (instead of registers), although none of them is shared between threads. 
    // This choice increases performance, particularly when the number of threads is large (§6.2.5).
    __shared__ T weights[BLOCK_DIM][3][3];

    int laneid = threadIdx.x & 0x1f;
    int cellid = -1;
    bool boundary;
    if (idx < n_particles) {
        cellid = grid_index[idx];
        boundary = (laneid == 0) || cellid != grid_index[idx - 1];
    }
    else {
        boundary = true;
    }
    uint32_t mark = __ballot_sync(0xFFFFFFFF, boundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
    mark = interval;
    #pragma unroll
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
        mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    }
    mark = __shfl_sync(0xFFFFFFFF, mark, 0);
    __syncthreads();

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(positions[idx * 3 + 0] * config::G_DX_INV - 0.5),
            static_cast<uint32_t>(positions[idx * 3 + 1] * config::G_DX_INV - 0.5),
            static_cast<uint32_t>(positions[idx * 3 + 2] * config::G_DX_INV - 0.5)
        };
        T fx[3] = {
            positions[idx * 3 + 0] * config::G_DX_INV - static_cast<T>(base[0]),
            positions[idx * 3 + 1] * config::G_DX_INV - static_cast<T>(base[1]),
            positions[idx * 3 + 2] * config::G_DX_INV - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = 0.5 * (1.5 - fx[i]) * (1.5 - fx[i]);
            weights[threadIdx.x][1][i] = 0.75 - (fx[i] - 1.0) * (fx[i] - 1.0);
            weights[threadIdx.x][2][i] = 0.5 * (fx[i] - 0.5) * (fx[i] - 0.5);
        }
        
        const T* C = &affine_matrices[idx * 9];

        T mass = volumes[idx] * config::DENSITY;
        T vel[3] = {
            velocities[idx * 3 + 0],
            velocities[idx * 3 + 1],
            velocities[idx * 3 + 2]
        };
        T B[9];
        const T* stress = &taus[idx * 9];
        const T* force = &forces[idx * 3];

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            B[i] = (-dt * config::G_D_INV) * stress[i] + C[i] * mass;
        }

        T val[4];

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    T xi_minus_xp[3] = {
                        (i - fx[0]) * config::G_DX,
                        (j - fx[1]) * config::G_DX,
                        (k - fx[2]) * config::G_DX
                    };

                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];
                    val[0] = mass * weight;
                    val[1] = vel[0] * val[0];
                    val[2] = vel[1] * val[0];
                    val[3] = vel[2] * val[0];

                    // apply gravity
                    val[config::GRAVITY_AXIS + 1] += val[0] * config::GRAVITY * dt;

                    val[1] += (B[0] * xi_minus_xp[0] + B[1] * xi_minus_xp[1] + B[2] * xi_minus_xp[2]) * weight;
                    val[2] += (B[3] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] + B[5] * xi_minus_xp[2]) * weight;
                    val[3] += (B[6] * xi_minus_xp[0] + B[7] * xi_minus_xp[1] + B[8] * xi_minus_xp[2]) * weight;
                    val[1] += force[0] * dt * weight;
                    val[2] += force[1] * dt * weight;
                    val[3] += force[2] * dt * weight;

                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        T tmp[4]; for (int ii = 0; ii < 4; ++ii) tmp[ii] = __shfl_down_sync(0xFFFFFFFF, val[ii], iter);
                        if (interval >= iter) for (int ii = 0; ii < 4; ++ii) val[ii] += tmp[ii];
                    }

                    if (boundary) {
                        const uint32_t target_cell_index = cell_index(base[0] + i, base[1] + j, base[2] + k);
                        const uint32_t target_grid_index = target_cell_index >> (config::G_BLOCK_BITS * 3);
                        g_touched_flags[target_grid_index] = 1;
                        atomicAdd(&(g_masses[target_cell_index]), val[0]);
                        atomicAdd(&(g_momentum[target_cell_index * 3 + 0]), val[1]);
                        atomicAdd(&(g_momentum[target_cell_index * 3 + 1]), val[2]);
                        atomicAdd(&(g_momentum[target_cell_index * 3 + 2]), val[3]);
                    }
                }
            }
        }
    }
}

template<typename T, int BLOCK_DIM>
__global__ void gather_touched_grid_kernel(
    const uint32_t* g_touched_flags,
    uint32_t* g_touched_ids,
    uint32_t* g_touched_cnt,
    T* g_masses // NOTE (changyu): placeholder here to avoid template re-instantialization error
) {
    __shared__ uint32_t shared_touched_ids[BLOCK_DIM];

    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;

    uint32_t touched_count = 0;
    uint32_t touched_idx = 0;

    if (idx < config::G_GRID_VOLUME && g_touched_flags[idx]) {
        touched_idx = idx;
        touched_count = 1;
    }

    uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, touched_count);
    uint32_t warp_pos = __popc(warp_mask & ((1U << lane_id) - 1));

    if (touched_count) {
        shared_touched_ids[warp_id * warpSize + warp_pos] = touched_idx;
    }

    if (lane_id == 0) {
        uint32_t block_offset = atomicAdd(g_touched_cnt, __popc(warp_mask));
        shared_touched_ids[warp_id * warpSize + 31] = block_offset;
    }
    __syncwarp();

    uint32_t global_offset = shared_touched_ids[warp_id * warpSize + 31];
    if (touched_count) {
        g_touched_ids[global_offset + warp_pos] = touched_idx;
    }
}

template<typename T>
__global__ void clean_grid_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    uint32_t* g_touched_flags,
    T* g_masses,
    T* g_momentum) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        g_touched_flags[block_idx] = 0;
        g_masses[cell_idx] = 0;
        g_momentum[cell_idx * 3 + 0] = 0;
        g_momentum[cell_idx * 3 + 1] = 0;
        g_momentum[cell_idx * 3 + 2] = 0;
    }
}

template<typename T>
__global__ void update_grid_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    T* g_masses,
    T* g_momentum) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        if (g_masses[cell_idx] > 0.) {
            // printf("m=%lf mv=(%lf %lf %lf)\n", g_masses[cell_idx], g_momentum[cell_idx * 3 + 0], g_momentum[cell_idx * 3 + 1], g_momentum[cell_idx * 3 + 2]);
            g_momentum[cell_idx * 3 + 0] /= g_masses[cell_idx];
            g_momentum[cell_idx * 3 + 1] /= g_masses[cell_idx];
            g_momentum[cell_idx * 3 + 2] /= g_masses[cell_idx];

            // apply boundary condition
            const int boundary_condition  = static_cast<int>(std::floor(config::G_BOUNDARY_CONDITION));
            uint3 xyz = inverse_cell_index(cell_idx);
            if (xyz.x < boundary_condition && g_momentum[cell_idx * 3 + 0] < 0) g_momentum[cell_idx * 3 + 0] = 0;
            if (xyz.x >= config::G_DOMAIN_SIZE - boundary_condition && g_momentum[cell_idx * 3 + 0] > 0) g_momentum[cell_idx * 3 + 0] = 0;
            if (xyz.y < boundary_condition && g_momentum[cell_idx * 3 + 1] < 0) g_momentum[cell_idx * 3 + 1] = 0;
            if (xyz.y >= config::G_DOMAIN_SIZE - boundary_condition && g_momentum[cell_idx * 3 + 1] > 0) g_momentum[cell_idx * 3 + 1] = 0;
            if (xyz.z < boundary_condition && g_momentum[cell_idx * 3 + 2] < 0) g_momentum[cell_idx * 3 + 2] = 0;
            if (xyz.z >= config::G_DOMAIN_SIZE - boundary_condition && g_momentum[cell_idx * 3 + 2] > 0) g_momentum[cell_idx * 3 + 2] = 0;
        
            // TODO, NOTE (changyu): ad-hoc hack for a sphere sdf
            {
                T pos[3] = {
                    (xyz.x + T(.5)) * config::G_DX,
                    (xyz.y + T(.5)) * config::G_DX,
                    (xyz.z + T(.5)) * config::G_DX
                };
                
                const T sphere_radius = 0.08;
                const T sphere_pos[3] = { 0.5, 0.5, 0.5 };
                const T sphere_vel[3] = { 0., 0., 0. };
                const bool fixed = true;

                T dist = distance<3>(pos, sphere_pos) - sphere_radius;
                T normal[3] = { 
                    (pos[0] - sphere_pos[0]) / (dist + T(1e-10)),
                    (pos[1] - sphere_pos[1]) / (dist + T(1e-10)),
                    (pos[2] - sphere_pos[2]) / (dist + T(1e-10))
                };

                bool inside = false;
                T dotnv = T(0.);
                T diff_vel[3] = { T(0.), T(0.), T(0.) };
                if (dist < 0.) {
                    diff_vel[0] = sphere_vel[0] - g_momentum[cell_idx * 3 + 0];
                    diff_vel[1] = sphere_vel[1] - g_momentum[cell_idx * 3 + 1];
                    diff_vel[2] = sphere_vel[2] - g_momentum[cell_idx * 3 + 2];
                    dotnv = dot<3>(normal, diff_vel);
                    if (dotnv > 0. || fixed) {
                        inside = true;
                    }
                }

                // NOTE (changyu): fixed, inside, dotnv, diff_vel, n = self.sdf.check(pos, vel)
                if (inside) {
                    if (fixed) {
                        g_momentum[cell_idx * 3 + 0] = 0.;
                        g_momentum[cell_idx * 3 + 1] = 0.;
                        g_momentum[cell_idx * 3 + 2] = 0.;
                    } else {
                        T dotnv_frac = dotnv * (1. - config::SDF_FRICTION);
                        g_momentum[cell_idx * 3 + 0] += diff_vel[0] * config::SDF_FRICTION + normal[0] * dotnv_frac;
                        g_momentum[cell_idx * 3 + 1] += diff_vel[1] * config::SDF_FRICTION + normal[1] * dotnv_frac;
                        g_momentum[cell_idx * 3 + 2] += diff_vel[2] * config::SDF_FRICTION + normal[2] * dotnv_frac;
                    }
                }
            }
        }
    }
}

template<typename T, int BLOCK_DIM>
__global__ void grid_to_particle_kernel(const size_t n_particles,
    T* positions, 
    T* velocities,
    T* affine_matrices,
    const T* g_momentum,
    const T dt) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // In [Fei et.al 2021],
    // we spill the B-spline weights (nine floats for each thread) by storing them into the shared memory
    // (instead of registers), although none of them is shared between threads. 
    // This choice increases performance, particularly when the number of threads is large (§6.2.5).
    __shared__ T weights[BLOCK_DIM][3][3];

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(positions[idx * 3 + 0] * config::G_DX_INV - 0.5),
            static_cast<uint32_t>(positions[idx * 3 + 1] * config::G_DX_INV - 0.5),
            static_cast<uint32_t>(positions[idx * 3 + 2] * config::G_DX_INV - 0.5)
        };
        T fx[3] = {
            positions[idx * 3 + 0] * config::G_DX_INV - static_cast<T>(base[0]),
            positions[idx * 3 + 1] * config::G_DX_INV - static_cast<T>(base[1]),
            positions[idx * 3 + 2] * config::G_DX_INV - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = 0.5 * (1.5 - fx[i]) * (1.5 - fx[i]);
            weights[threadIdx.x][1][i] = 0.75 - (fx[i] - 1.0) * (fx[i] - 1.0);
            weights[threadIdx.x][2][i] = 0.5 * (fx[i] - 0.5) * (fx[i] - 0.5);
        }

        T new_v[3];
        T new_C[9];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            new_v[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            new_C[i] = 0;
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    T xi_minus_xp[3] = {
                        (i - fx[0]),
                        (j - fx[1]),
                        (k - fx[2])
                    };

                    const uint32_t target_cell_index = cell_index(base[0] + i, base[1] + j, base[2] + k);
                    const T* g_v = &g_momentum[target_cell_index * 3];

                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];
                    new_v[0] += weight * g_v[0];
                    new_v[1] += weight * g_v[1];
                    new_v[2] += weight * g_v[2];
                    // printf("weight=%lf, g_v=(%lf %lf %lf)\n", weight, g_v[0], g_v[1], g_v[2]);

                    // printf("i=%d j=%d k=%d\n", i, j, k);
                    // printf("weight=%.8lf\n", weight);
                    // printf("g_v=[%.8lf   %.8lf   %.8lf]\n", g_v[0], g_v[1], g_v[2]);
                    // printf("xip=[%.8lf   %.8lf   %.8lf]\n", xi_minus_xp[0], xi_minus_xp[1], xi_minus_xp[2]);
                    new_C[0] += 4 * config::G_DX_INV * weight * g_v[0] * xi_minus_xp[0];
                    new_C[1] += 4 * config::G_DX_INV * weight * g_v[0] * xi_minus_xp[1];
                    new_C[2] += 4 * config::G_DX_INV * weight * g_v[0] * xi_minus_xp[2];
                    new_C[3] += 4 * config::G_DX_INV * weight * g_v[1] * xi_minus_xp[0];
                    new_C[4] += 4 * config::G_DX_INV * weight * g_v[1] * xi_minus_xp[1];
                    new_C[5] += 4 * config::G_DX_INV * weight * g_v[1] * xi_minus_xp[2];
                    new_C[6] += 4 * config::G_DX_INV * weight * g_v[2] * xi_minus_xp[0];
                    new_C[7] += 4 * config::G_DX_INV * weight * g_v[2] * xi_minus_xp[1];
                    new_C[8] += 4 * config::G_DX_INV * weight * g_v[2] * xi_minus_xp[2];
                }
            }
        }

        velocities[idx * 3 + 0] = new_v[0];
        velocities[idx * 3 + 1] = new_v[1];
        velocities[idx * 3 + 2] = new_v[2];
        affine_matrices[idx * 9 + 0] = new_C[0];
        affine_matrices[idx * 9 + 1] = new_C[1];
        affine_matrices[idx * 9 + 2] = new_C[2];
        affine_matrices[idx * 9 + 3] = new_C[3];
        affine_matrices[idx * 9 + 4] = new_C[4];
        affine_matrices[idx * 9 + 5] = new_C[5];
        affine_matrices[idx * 9 + 6] = new_C[6];
        affine_matrices[idx * 9 + 7] = new_C[7];
        affine_matrices[idx * 9 + 8] = new_C[8];

        // Advection
        positions[idx * 3 + 0] += new_v[0] * dt;
        positions[idx * 3 + 1] += new_v[1] * dt;
        positions[idx * 3 + 2] += new_v[2] * dt;

        // printf("v=\n");
        // printf("[%.8lf   %.8lf   %.8lf]\n", new_v[0], new_v[1], new_v[2]);
        // printf("C=\n");
        // printf("[[%.8lf   %.8lf   %.8lf] \n", new_C[0], new_C[1], new_C[2]);
        // printf(" [%.8lf   %.8lf   %.8lf] \n", new_C[3], new_C[4], new_C[5]);
        // printf(" [%.8lf   %.8lf   %.8lf]]\n", new_C[6], new_C[7], new_C[8]);
    }
}

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake