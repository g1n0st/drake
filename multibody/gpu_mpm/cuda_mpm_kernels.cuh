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
            positions[idx * 3 + i] = (positions[v0 * 3 + i] + positions[v1 * 3 + i] + positions[v2 * 3 + i]) / T(3.);
            velocities[idx * 3 + i] = (velocities[v0 * 3 + i] + velocities[v1 * 3 + i] + velocities[v2 * 3 + i]) / T(3.);
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
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            F[i] = Q[i];
        }

        T D0xD1[3];
        cross_product3(D0, D1, D0xD1);
        T volume_4 = norm<3>(D0xD1) / T(8.) * config::G_DX<T>;

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
    dphi_dF[0] = T(2.) * config::MU<T> * (F[0] - R[0]) + config::LAMBDA<T> * (J - T(1.)) * J * Finv[0];
    dphi_dF[1] = T(2.) * config::MU<T> * (F[1] - R[1]) + config::LAMBDA<T> * (J - T(1.)) * J * Finv[2];
    dphi_dF[2] = T(2.) * config::MU<T> * (F[2] - R[2]) + config::LAMBDA<T> * (J - T(1.)) * J * Finv[1];
    dphi_dF[3] = T(2.) * config::MU<T> * (F[3] - R[3]) + config::LAMBDA<T> * (J - T(1.)) * J * Finv[3];
}

template<typename T>
__device__ __host__
inline void compute_dphi_dF(const T* F, T* dphi_dF) {
    // A00=0, A01=1, A02=2
    // A10=3, A11=4, A12=5
    // A20=6, A21=7, A22=8
    T Q[9], R[9];
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
    T g = config::GAMMA<T> * rr;
    T gp = config::GAMMA<T>;
    T fp = 0;
    if (R[8] < T(1.)) {
        fp = -config::K<T> * (T(1.) - R[8]) * (T(1.) - R[8]);
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
    if (config::GAMMA<T> == T(0.)) { // CASE 1: no friction
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
        const T gamma_over_k = config::GAMMA<T> / config::K<T>;
        T zz = config::c_F<T> * (R[8] - T(1.)) * (R[8] - T(1.));
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
    const int* index_mappings,
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
    int face_pid = index_mappings[idx];
    if (idx < n_faces) {
        int v0 = index_mappings[indices[idx * 3 + 0]];
        int v1 = index_mappings[indices[idx * 3 + 1]];
        int v2 = index_mappings[indices[idx * 3 + 2]];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            positions[face_pid * 3 + i] = (positions[v0 * 3 + i] + positions[v1 * 3 + i] + positions[v2 * 3 + i]) / T(3.);
            velocities[face_pid * 3 + i] = (velocities[v0 * 3 + i] + velocities[v1 * 3 + i] + velocities[v2 * 3 + i]) / T(3.);
        }

        T* F = &deformation_gradients[idx * 9];
        const T* C = &affine_matrices[face_pid * 9];
        T ctF[9]; // cotangent F

        // Eq.4 in Jiang et.al 2017, dE_p, β(x̂) = (∇x̂)p dE,n_p, β
        // but we could reuse traditional MPM deformation gradient updated as:
        // F̂E_p(x̂) = (∇x̂)p FE,n_p
        ctF[0] = F[0];
        ctF[1] = F[1];
        ctF[2] = (T(1.0) + dt * C[0]) * F[2] + dt * C[1] * F[5] + dt * C[2] * F[8];

        ctF[3] = F[3];
        ctF[4] = F[4];
        ctF[5] = dt * C[3] * F[2] + (T(1.0) + dt * C[4]) * F[5] + dt * C[5] * F[8];

        ctF[6] = F[6];
        ctF[7] = F[7];
        ctF[8] = dt * C[6] * F[2] + dt * C[7] * F[5] + (T(1.0) + dt * C[8]) * F[8];

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
            VP_local[i] *= volumes[face_pid];
        }

        // technical document .(15) part 2
        T VP_local_c2[3] = { VP_local[2], VP_local[5], VP_local[8] };
        T ctF_c2[3] = { ctF[2], ctF[5], ctF[8] };
        outer_product<3, T>(VP_local_c2, ctF_c2, &taus[face_pid * 9]);

        T grad_N_hat[6] = {
            T(-1.), T(1.), T(0.),
            T(-1.), T(0.), T(1.)
        };
        T grad_N[6];
        T Dm_inverse_T[4];
        transpose<2, 2, T>(Dm_inverse, Dm_inverse_T);
        matmul<2, 2, 3, T>(Dm_inverse_T, grad_N_hat, grad_N);
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
        uint32_t xi = static_cast<uint32_t>(x * config::G_DX_INV<T> - T(0.5));
        uint32_t yi = static_cast<uint32_t>(y * config::G_DX_INV<T> - T(0.5));
        uint32_t zi = static_cast<uint32_t>(z * config::G_DX_INV<T> - T(0.5));
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
    const int* current_pids,
    const uint32_t* next_sort_ids,
    T* next_positions,
    T* next_velocities,
    T* next_volumes,
    T* next_affine_matrices,
    int* next_pids,
    int* index_mappings
    ) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        next_volumes[idx] = current_volumes[next_sort_ids[idx]];
        next_pids[idx] = current_pids[next_sort_ids[idx]];
        index_mappings[next_pids[idx]] = idx;

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            next_positions[idx * 3 + i] = current_positions[next_sort_ids[idx] * 3 + i];
            next_velocities[idx * 3 + i] = current_velocities[next_sort_ids[idx] * 3 + i];
        }

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            next_affine_matrices[idx * 9 + i] = current_affine_matrices[next_sort_ids[idx] * 9 + i];
        }
    }
}

template<typename T, int BLOCK_DIM, bool APPLY_GRAVITY=true, bool APPLY_ELASTICITY=true>
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

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(positions[idx * 3 + 0] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 1] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 2] * config::G_DX_INV<T> - T(0.5))
        };
        T fx[3] = {
            positions[idx * 3 + 0] * config::G_DX_INV<T> - static_cast<T>(base[0]),
            positions[idx * 3 + 1] * config::G_DX_INV<T> - static_cast<T>(base[1]),
            positions[idx * 3 + 2] * config::G_DX_INV<T> - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        const T mass = volumes[idx] * config::DENSITY<T>;
        const T* vel = &velocities[idx * 3];

        T B[9];
        const T* C = &affine_matrices[idx * 9];
        const T* stress = &taus[idx * 9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            if constexpr (APPLY_ELASTICITY) {
                B[i] = (-dt * config::G_D_INV<T>) * stress[i] + C[i] * mass;
            } else {
                B[i] = C[i] * mass;
            }
        }

        T val[4];

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    T xi_minus_xp[3] = {
                        (i - fx[0]) * config::G_DX<T>,
                        (j - fx[1]) * config::G_DX<T>,
                        (k - fx[2]) * config::G_DX<T>
                    };

                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];

                    val[0] = mass * weight;
                    val[1] = vel[0] * val[0];
                    val[2] = vel[1] * val[0];
                    val[3] = vel[2] * val[0];
                    // apply gravity
                    if constexpr (APPLY_GRAVITY) {
                        val[config::GRAVITY_AXIS + 1] += val[0] * config::GRAVITY<T> * dt;
                    }

                    val[1] += (B[0] * xi_minus_xp[0] + B[1] * xi_minus_xp[1] + B[2] * xi_minus_xp[2]) * weight;
                    val[2] += (B[3] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] + B[5] * xi_minus_xp[2]) * weight;
                    val[3] += (B[6] * xi_minus_xp[0] + B[7] * xi_minus_xp[1] + B[8] * xi_minus_xp[2]) * weight;
                    if constexpr (APPLY_ELASTICITY) {
                        const T* force = &forces[idx * 3];
                        val[1] += force[0] * dt * weight;
                        val[2] += force[1] * dt * weight;
                        val[3] += force[2] * dt * weight;
                    }

                    for (int iter = 1; iter <= mark; iter <<= 1) {
                        T tmp[4]; 
                        #pragma unroll
                        for (int ii = 0; ii < 4; ++ii) tmp[ii] = __shfl_down_sync(0xFFFFFFFF, val[ii], iter);
                        if (interval >= iter) {
                            #pragma unroll
                            for (int ii = 0; ii < 4; ++ii) val[ii] += tmp[ii];
                        }
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
    const uint32_t* g_touched_ids,
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
__global__ void clean_grid_contact_kernel(
    const uint32_t touched_cells_cnt,
    const uint32_t* g_touched_ids,
    T* g_Hess,
    T* g_Grad,
    T* g_Dir,
    T* g_alpha,
    T* g_E0,
    T* g_E1) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        g_alpha[cell_idx] = T(-1.);
        g_Grad[cell_idx * 3 + 0] = 0;
        g_Grad[cell_idx * 3 + 1] = 0;
        g_Grad[cell_idx * 3 + 2] = 0;
        g_Dir[cell_idx * 3 + 0] = 0;
        g_Dir[cell_idx * 3 + 1] = 0;
        g_Dir[cell_idx * 3 + 2] = 0;
        #pragma unroll
        for (int i = 0; i < 9; ++i) g_Hess[cell_idx * 9 + i] = 0;
        g_E0[cell_idx] = 0;
        g_E1[cell_idx] = 0;
    }
}

template<typename T, int MPM_BOUNDARY_CONDITION=-1>
__global__ void update_grid_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    T* g_masses,
    T* g_momentum,
    T* g_v_star) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        if (g_masses[cell_idx] > T(0.)) {
            T *g_vel = &g_momentum[cell_idx * 3];
            // printf("m=%lf mv=(%lf %lf %lf)\n", g_masses[cell_idx], g_vel[0], g_vel[1], g_vel[2]);
            g_vel[0] /= g_masses[cell_idx];
            g_vel[1] /= g_masses[cell_idx];
            g_vel[2] /= g_masses[cell_idx];

            // apply boundary condition
            const int boundary_condition = config::G_BOUNDARY_CONDITION;
            uint3 xyz = inverse_cell_index(cell_idx);
            if (xyz.x < boundary_condition && g_vel[0] < 0) g_vel[0] = 0;
            if (xyz.x >= config::G_DOMAIN_SIZE - boundary_condition && g_vel[0] > 0) g_vel[0] = 0;
            if (xyz.y < boundary_condition && g_vel[1] < 0) g_vel[1] = 0;
            if (xyz.y >= config::G_DOMAIN_SIZE - boundary_condition && g_vel[1] > 0) g_vel[1] = 0;
            if (xyz.z < boundary_condition && g_vel[2] < 0) g_vel[2] = 0;
            if (xyz.z >= config::G_DOMAIN_SIZE - boundary_condition && g_vel[2] > 0) g_vel[2] = 0;

            {
                T pos[3] = {
                    (xyz.x + T(.5)) * config::G_DX<T>,
                    (xyz.y + T(.5)) * config::G_DX<T>,
                    (xyz.z + T(.5)) * config::G_DX<T>
                };
                bool fixed = false;
                bool inside = false;
                T dist = T(0);
                T diff_vel[3] = { T(0), T(0), T(0) };
                T normal[3] = { T(0), T(0), T(0) };
                T dotnv = T(0);

                if constexpr (MPM_BOUNDARY_CONDITION == 0) {
                    const T sphere_radius = T(0.08);
                    const T sphere_pos[3] = { T(0.5), T(0.5), T(0.5) };
                    const T sphere_vel[3] = { T(0.), T(0.), T(0.) };

                    dist = distance<3>(pos, sphere_pos) - sphere_radius;
                    normal[0] = (pos[0] - sphere_pos[0]);
                    normal[1] = (pos[1] - sphere_pos[1]);
                    normal[2] = (pos[2] - sphere_pos[2]);
                    normalize<3, T>(normal);
                    if (dist < T(0.)) {
                        diff_vel[0] = sphere_vel[0] - g_vel[0];
                        diff_vel[1] = sphere_vel[1] - g_vel[1];
                        diff_vel[2] = sphere_vel[2] - g_vel[2];
                        dotnv = dot<3>(normal, diff_vel);
                        if (dotnv > T(0.) || fixed) {
                            inside = true;
                        }
                    }
                }

                else if constexpr (MPM_BOUNDARY_CONDITION == 1) {
                    const T sphere_radius = T(0.04);
                    const T sphere_pos1[3] = { T(0.38), T(0.38), T(0.75) };
                    const T sphere_pos2[3] = { T(0.38), T(0.62), T(0.75) };
                    const T sphere_vel[3] = { T(0.), T(0.), T(0.) };
                    fixed = true;

                    dist = distance<3>(pos, sphere_pos1) - sphere_radius;
                    normal[0] = (pos[0] - sphere_pos1[0]);
                    normal[1] = (pos[1] - sphere_pos1[1]);
                    normal[2] = (pos[2] - sphere_pos1[2]);
                    normalize<3, T>(normal);

                    if (dist < T(0.)) {
                        diff_vel[0] = sphere_vel[0] - g_vel[0];
                        diff_vel[1] = sphere_vel[1] - g_vel[1];
                        diff_vel[2] = sphere_vel[2] - g_vel[2];
                        dotnv = dot<3>(normal, diff_vel);
                        if (dotnv > T(0.) || fixed) {
                            inside = true;
                        }
                    }
                    else {
                        dist = distance<3>(pos, sphere_pos2) - sphere_radius;
                        normal[0] = (pos[0] - sphere_pos2[0]);
                        normal[1] = (pos[1] - sphere_pos2[1]);
                        normal[2] = (pos[2] - sphere_pos2[2]);
                        normalize<3, T>(normal);

                        dotnv = T(0.);
                        if (dist < T(0.)) {
                            diff_vel[0] = sphere_vel[0] - g_vel[0];
                            diff_vel[1] = sphere_vel[1] - g_vel[1];
                            diff_vel[2] = sphere_vel[2] - g_vel[2];
                            dotnv = dot<3>(normal, diff_vel);
                            if (dotnv > T(0.) || fixed) {
                                inside = true;
                            }
                        }
                    }
                }

                // z-axis=0.1 used for cloth/tshirt folding demos
                else if constexpr (MPM_BOUNDARY_CONDITION == 2) {
                    normal[0] = T(0.);
                    normal[1] = T(0.);
                    normal[2] = T(1.);
                    dist = pos[2] - T(0.11);
                    if (dist < 0) {
                        inside = true;
                        diff_vel[0] = -g_vel[0];
                        diff_vel[1] = -g_vel[1];
                        diff_vel[2] = -g_vel[2];
                        dotnv = dot<3>(diff_vel, normal);
                    }
                }

                // four-corner suspension used for bagging demo
                else if constexpr (MPM_BOUNDARY_CONDITION == 3) {
                    fixed = true;
                    const T sphere_radius = T(0.02);
                    const T span[2] = {T(0.3), T(0.7)};

                    for (int _ = 0; _ < 4; ++_) {
                        const T sphere_pos[3] = {span[_%2], span[_/2], 0.5};
                        dist = distance<3>(pos, sphere_pos) - sphere_radius;
                        normal[0] = (pos[0] - sphere_pos[0]);
                        normal[1] = (pos[1] - sphere_pos[1]);
                        normal[2] = (pos[2] - sphere_pos[2]);
                        normalize<3, T>(normal);

                        if (dist < T(0.)) {
                            diff_vel[0] = -g_vel[0];
                            diff_vel[1] = -g_vel[1];
                            diff_vel[2] = -g_vel[2];
                            dotnv = dot<3>(normal, diff_vel);
                            inside = true;
                            break;
                        }
                    }
                }

                // NOTE (changyu): fixed, inside, dotnv, diff_vel, n = self.sdf.check(pos, vel)
                if (inside) {
                    if (fixed) {
                        g_vel[0] += diff_vel[0];
                        g_vel[1] += diff_vel[1];
                        g_vel[2] += diff_vel[2];
                    } else {
                        T dotnv_frac = dotnv * (1. - config::SDF_FRICTION<T>);
                        g_vel[0] += diff_vel[0] * config::SDF_FRICTION<T> + normal[0] * dotnv_frac;
                        g_vel[1] += diff_vel[1] * config::SDF_FRICTION<T> + normal[1] * dotnv_frac;
                        g_vel[2] += diff_vel[2] * config::SDF_FRICTION<T> + normal[2] * dotnv_frac;
                    }
                }
            }

            g_v_star[cell_idx * 3 + 0] = g_vel[0];
            g_v_star[cell_idx * 3 + 1] = g_vel[1];
            g_v_star[cell_idx * 3 + 2] = g_vel[2];
        }
    }
}

template<typename T, int BLOCK_DIM, bool CONTACT_TRANSFER, bool POST_CONTACT>
__global__ void grid_to_particle_kernel(const size_t n_particles,
    T* positions, 
    T* velocities,
    T* affine_matrices,
    const T* g_masses,
    const T* g_momentum,
    const T* g_v_star,
    const T dt) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // In [Fei et.al 2021],
    // we spill the B-spline weights (nine floats for each thread) by storing them into the shared memory
    // (instead of registers), although none of them is shared between threads. 
    // This choice increases performance, particularly when the number of threads is large (§6.2.5).
    __shared__ T weights[BLOCK_DIM][3][3];

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(positions[idx * 3 + 0] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 1] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 2] * config::G_DX_INV<T> - T(0.5))
        };
        T fx[3] = {
            positions[idx * 3 + 0] * config::G_DX_INV<T> - static_cast<T>(base[0]),
            positions[idx * 3 + 1] * config::G_DX_INV<T> - static_cast<T>(base[1]),
            positions[idx * 3 + 2] * config::G_DX_INV<T> - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        T old_v[3];
        T new_v[3];
        T new_C[9], new_CT[9];
        T old_C[9], old_CT[9];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            new_v[i] = 0;
            old_v[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            new_C[i] = 0;
            old_C[i] = 0;
        }

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    T xi_minus_xp[3] = {
                        (i - fx[0]),
                        (j - fx[1]),
                        (k - fx[2])
                    };

                    const uint32_t target_cell_index = cell_index(base[0] + i, base[1] + j, base[2] + k);
                    const T* g_v = &g_momentum[target_cell_index * 3];

                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];

                    if constexpr (CONTACT_TRANSFER) {
                        const T &g_m = g_masses[target_cell_index];
                        if (g_m > T(1e-7)) {
                            new_v[0] += weight * g_v[0];
                            new_v[1] += weight * g_v[1];
                            new_v[2] += weight * g_v[2];
                        }
                    } else {
                        new_v[0] += weight * g_v[0];
                        new_v[1] += weight * g_v[1];
                        new_v[2] += weight * g_v[2];
                        if constexpr (POST_CONTACT) {
                            old_v[0] += weight * g_v_star[target_cell_index * 3 + 0];
                            old_v[1] += weight * g_v_star[target_cell_index * 3 + 1];
                            old_v[2] += weight * g_v_star[target_cell_index * 3 + 2];
                        }
                        // printf("weight=%lf, g_v=(%lf %lf %lf)\n", weight, g_v[0], g_v[1], g_v[2]);

                        // printf("i=%d j=%d k=%d\n", i, j, k);
                        // printf("weight=%.8lf\n", weight);
                        // printf("g_v=[%.8lf   %.8lf   %.8lf]\n", g_v[0], g_v[1], g_v[2]);
                        // printf("xip=[%.8lf   %.8lf   %.8lf]\n", xi_minus_xp[0], xi_minus_xp[1], xi_minus_xp[2]);
                        new_C[0] += 4 * config::G_DX_INV<T> * weight * g_v[0] * xi_minus_xp[0];
                        new_C[1] += 4 * config::G_DX_INV<T> * weight * g_v[0] * xi_minus_xp[1];
                        new_C[2] += 4 * config::G_DX_INV<T> * weight * g_v[0] * xi_minus_xp[2];
                        new_C[3] += 4 * config::G_DX_INV<T> * weight * g_v[1] * xi_minus_xp[0];
                        new_C[4] += 4 * config::G_DX_INV<T> * weight * g_v[1] * xi_minus_xp[1];
                        new_C[5] += 4 * config::G_DX_INV<T> * weight * g_v[1] * xi_minus_xp[2];
                        new_C[6] += 4 * config::G_DX_INV<T> * weight * g_v[2] * xi_minus_xp[0];
                        new_C[7] += 4 * config::G_DX_INV<T> * weight * g_v[2] * xi_minus_xp[1];
                        new_C[8] += 4 * config::G_DX_INV<T> * weight * g_v[2] * xi_minus_xp[2];
                        if constexpr (POST_CONTACT) {
                            old_C[0] += 4 * config::G_DX_INV<T> * weight * g_v_star[0] * xi_minus_xp[0];
                            old_C[1] += 4 * config::G_DX_INV<T> * weight * g_v_star[0] * xi_minus_xp[1];
                            old_C[2] += 4 * config::G_DX_INV<T> * weight * g_v_star[0] * xi_minus_xp[2];
                            old_C[3] += 4 * config::G_DX_INV<T> * weight * g_v_star[1] * xi_minus_xp[0];
                            old_C[4] += 4 * config::G_DX_INV<T> * weight * g_v_star[1] * xi_minus_xp[1];
                            old_C[5] += 4 * config::G_DX_INV<T> * weight * g_v_star[1] * xi_minus_xp[2];
                            old_C[6] += 4 * config::G_DX_INV<T> * weight * g_v_star[2] * xi_minus_xp[0];
                            old_C[7] += 4 * config::G_DX_INV<T> * weight * g_v_star[2] * xi_minus_xp[1];
                            old_C[8] += 4 * config::G_DX_INV<T> * weight * g_v_star[2] * xi_minus_xp[2];
                        }
                    }
                }
            }
        }

        if constexpr (CONTACT_TRANSFER) {
            velocities[idx * 3 + 0] = new_v[0];
            velocities[idx * 3 + 1] = new_v[1];
            velocities[idx * 3 + 2] = new_v[2];
        } else {
            velocities[idx * 3 + 0] = new_v[0];
            velocities[idx * 3 + 1] = new_v[1];
            velocities[idx * 3 + 2] = new_v[2];

            transpose<3, 3, T>(new_C, new_CT);
            // Advection
            if constexpr (POST_CONTACT) {
                transpose<3, 3, T>(old_C, old_CT);

                affine_matrices[idx * 9 + 0] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[0] - old_C[0]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[0] - old_CT[0]);
                affine_matrices[idx * 9 + 1] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[1] - old_C[1]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[1] - old_CT[1]);
                affine_matrices[idx * 9 + 2] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[2] - old_C[2]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[2] - old_CT[2]);
                affine_matrices[idx * 9 + 3] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[3] - old_C[3]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[3] - old_CT[3]);
                affine_matrices[idx * 9 + 4] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[4] - old_C[4]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[4] - old_CT[4]);
                affine_matrices[idx * 9 + 5] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[5] - old_C[5]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[5] - old_CT[5]);
                affine_matrices[idx * 9 + 6] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[6] - old_C[6]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[6] - old_CT[6]);
                affine_matrices[idx * 9 + 7] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[7] - old_C[7]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[7] - old_CT[7]);
                affine_matrices[idx * 9 + 8] = ((config::V<T> + T(1.)) * T(.5)) * (new_C[8] - old_C[8]) + ((config::V<T> - T(1.)) * T(.5)) * (new_CT[8] - old_CT[8]);

                positions[idx * 3 + 0] += (new_v[0] - old_v[0]) * dt;
                positions[idx * 3 + 1] += (new_v[1] - old_v[1]) * dt;
                positions[idx * 3 + 2] += (new_v[2] - old_v[2]) * dt;
            }
            else {
                affine_matrices[idx * 9 + 0] = ((config::V<T> + T(1.)) * T(.5)) * new_C[0] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[0];
                affine_matrices[idx * 9 + 1] = ((config::V<T> + T(1.)) * T(.5)) * new_C[1] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[1];
                affine_matrices[idx * 9 + 2] = ((config::V<T> + T(1.)) * T(.5)) * new_C[2] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[2];
                affine_matrices[idx * 9 + 3] = ((config::V<T> + T(1.)) * T(.5)) * new_C[3] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[3];
                affine_matrices[idx * 9 + 4] = ((config::V<T> + T(1.)) * T(.5)) * new_C[4] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[4];
                affine_matrices[idx * 9 + 5] = ((config::V<T> + T(1.)) * T(.5)) * new_C[5] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[5];
                affine_matrices[idx * 9 + 6] = ((config::V<T> + T(1.)) * T(.5)) * new_C[6] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[6];
                affine_matrices[idx * 9 + 7] = ((config::V<T> + T(1.)) * T(.5)) * new_C[7] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[7];
                affine_matrices[idx * 9 + 8] = ((config::V<T> + T(1.)) * T(.5)) * new_C[8] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[8];

                positions[idx * 3 + 0] += new_v[0] * dt;
                positions[idx * 3 + 1] += new_v[1] * dt;
                positions[idx * 3 + 2] += new_v[2] * dt;
            }

            // printf("v=\n");
            // printf("[%.8lf   %.8lf   %.8lf]\n", new_v[0], new_v[1], new_v[2]);
            // printf("C=\n");
            // printf("[[%.8lf   %.8lf   %.8lf] \n", new_C[0], new_C[1], new_C[2]);
            // printf(" [%.8lf   %.8lf   %.8lf] \n", new_C[3], new_C[4], new_C[5]);
            // printf(" [%.8lf   %.8lf   %.8lf]]\n", new_C[6], new_C[7], new_C[8]);
        }
    }
}

template<typename T, int BLOCK_DIM>
__global__ void initialize_contact_velocities(const size_t n_contacts,
    T* contact_vel,
    const uint32_t* contact_mpm_id,
    const T* velocities0) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_contacts) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            contact_vel[idx * 3 + i] = velocities0[contact_mpm_id[idx] * 3 + i];
        }
    }
}

template<typename UINT>
__device__ __host__ inline UINT get_color_mask(UINT i, UINT j, UINT k) {
    return (i % 3U) * 9U + (j % 3U) * 3U + (k % 3U);
}

template<typename UINT>
__device__ __host__ inline void get_color_coordinates(UINT x, UINT y, UINT z, UINT color_mask, UINT &i, UINT &j, UINT &k) {
    UINT i_offset = color_mask / 9U;
    UINT j_offset = (color_mask % 9U) / 3U;
    UINT k_offset = color_mask % 3U;

    i = ((3U + i_offset) - (x % 3U)) % 3U;
    j = ((3U + j_offset) - (y % 3U)) % 3U;
    k = ((3U + k_offset) - (z % 3U)) % 3U;
}

template<typename T>
__device__ void compute_contact_grad_and_hess(
    const T phi0, const T dt, const T stiffness, const T damping, const T friction_mu, 
    const T *v0, const T *v_next,
    T *C_Hess, T *C_Grad) {
    /* Solves the contact problem for a single particle against a rigid body
        assuming the rigid body has infinite mass and inertia.

        Let phi be the penetration distance (positive when penetration occurs) and vn
        be the relative velocity of the particle with respect to the rigid body in the
        normal direction (vn>0 when separting). Then we have phi_dot = -vn.

        In the normal direction, the contact force is modeled as a linear elastic system
        with Hunt-Crossley dissipation.

        f = k * phi_+ * (1 + d * phi_dot)_+

        where phi_+ = max(0, phi)

        The momentum balance in the normal direction becomes

        m(vn_next - vn) = k * dt * (phi0 - dt * vn_next)_+ * (1 - d * vn_next)_+

        where we used the fact that phi = phi0 - dt * vn_next. This is a quadratic
        equation in vn_next, and we solve it to get the next velocity vn_next.

        The quadratic equation is ax^2 + bx + c = 0, where

        a = k * d * dt^2
        b = -m - (k * dt * (dt + d * phi0))
        c = k * dt * phi0 + m * vn

        After solving for vn_next, we check if the friction force lies in the friction
        cone, if not, we project the velocity back into the friction cone. */
    constexpr int kZAxis = 2;

    // NOTE: follow the pattern in https://github.com/RobotLocomotion/drake/blob/master/multibody/contact_solvers/sap/sap_hunt_crossley_constraint.cc
    // Check if predicted penetration is positive.
    // If not, then the contact force is not repulsive, don't apply it.
    const T xdot = -(v_next[kZAxis] - v0[kZAxis]);
    T phi = phi0 + dt * xdot;
    T damping_cond = T(1.) + damping * xdot;
    if (damping_cond <= 0 || phi <= 0) { // Quick exits
        #pragma unroll
        for (int i = 0; i < 9; ++i) C_Hess[i] = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i) C_Grad[i] = 0;
    }
    else {
        // normal component
        const T yn = stiffness * dt * (phi0 - dt * (v_next[kZAxis] - v0[kZAxis])) * (T(1.) - damping * (v_next[kZAxis] - v0[kZAxis])); // Eq. 13
        const T d2lndvn2 = stiffness * dt * (-dt - damping * phi0 + T(2.) * damping * dt * (v_next[kZAxis] - v0[kZAxis])); // Eq. 8

        // frictional component
        // For a physical model of compliance for which γn is only a function of vn
        const T yn0 = max(stiffness * dt * phi0, T(0.));
        const T ts_coeff = sqrt(v_next[0] * v_next[0] + v_next[1] * v_next[1] + config::epsv<T> * config::epsv<T>);
        const T ts_hat[2] = {v_next[0] / ts_coeff, v_next[1] / ts_coeff}; // Eq. 18

        const T yt[2] = {-friction_mu * yn0 * ts_hat[0], -friction_mu * yn0 * ts_hat[1]}; // Eq. 33
        T P_ts_hat[4];
        outer_product<2, T>(ts_hat, ts_hat, P_ts_hat);
        const T P_perp_ts_hat[4] = {
            T(1.) - P_ts_hat[0], -P_ts_hat[1],
            -P_ts_hat[2], T(1.) - P_ts_hat[3]
        };
        const T d2ltdvt2_coeff = -friction_mu * yn0 / ts_coeff; // Eq. 33, ts_coeff = ts_soft_norm + epsv
        const T d2ltdvt2[4] = {
            d2ltdvt2_coeff * P_perp_ts_hat[0],
            d2ltdvt2_coeff * P_perp_ts_hat[1],
            d2ltdvt2_coeff * P_perp_ts_hat[2],
            d2ltdvt2_coeff * P_perp_ts_hat[3]
        };

        C_Hess[0] = d2ltdvt2[0];
        C_Hess[1] = d2ltdvt2[1];
        C_Hess[2] = T(0.);

        C_Hess[3] = d2ltdvt2[2];
        C_Hess[4] = d2ltdvt2[3];
        C_Hess[5] = T(0.);

        C_Hess[6] = T(0.);
        C_Hess[7] = T(0.);
        C_Hess[8] = d2lndvn2;

        C_Grad[0] = yt[0];
        C_Grad[1] = yt[1];
        C_Grad[2] = yn;
    }
}

template<typename T, int BLOCK_DIM, bool JACOBI>
__global__ void contact_particle_to_grid_kernel(const size_t n_particles,
    const T* contact_pos,
    const T* contact_vel,
    const T* velocities,
    const T* volumes,
    const uint32_t* contact_mpm_id,
    const T* contact_dist,
    const T* contact_normal,
    const T* contact_rigid_v,
    const uint32_t* grid_index,
    T* g_Hess,
    T* g_Grad,
    const T dt,
    const T friction_mu,
    const T stiffness,
    const T damping,
    const uint32_t g_color_mask) {
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

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(contact_pos[idx * 3 + 0] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 1] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 2] * config::G_DX_INV<T> - T(0.5))
        };
        T fx[3] = {
            contact_pos[idx * 3 + 0] * config::G_DX_INV<T> - static_cast<T>(base[0]),
            contact_pos[idx * 3 + 1] * config::G_DX_INV<T> - static_cast<T>(base[1]),
            contact_pos[idx * 3 + 2] * config::G_DX_INV<T> - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        const T mass = volumes[contact_mpm_id[idx]] * config::DENSITY<T>;
        const T* particle_v0 = &velocities[contact_mpm_id[idx] * 3];
        const T* particle_v = &contact_vel[idx * 3];

        T nhat_W[3] = {-contact_normal[idx * 3 + 0], -contact_normal[idx * 3 + 1], -contact_normal[idx * 3 + 2]};
        // TODO (changyu): const GpuT phi0 = -(
        // static_cast<GpuT>(mpm_contact_pairs[i].penetration_distance) + 
        //     (mpm_state->positions_host()[mpm_contact_pairs[i].particle_in_contact_index] - 
        //     mpm_contact_pairs[i].particle_in_contact_position.template cast<GpuT>()).dot(nhat_W)
        //   );
        T phi0 = -contact_dist[idx];

        T v0_rel[3] = {
            particle_v0[0] - contact_rigid_v[idx * 3 + 0],
            particle_v0[1] - contact_rigid_v[idx * 3 + 1],
            particle_v0[2] - contact_rigid_v[idx * 3 + 2]
        };
        T v_rel[3] = {
            particle_v[0] - contact_rigid_v[idx * 3 + 0],
            particle_v[1] - contact_rigid_v[idx * 3 + 1],
            particle_v[2] - contact_rigid_v[idx * 3 + 2]
        };

        constexpr int kZAxis = 2;
        T R_WC[9], R_CW[9]; // for each contact pair, Ji = R_CWp * wip
        make_from_one_unit_vector(nhat_W, kZAxis, R_WC);
        transpose<3, 3, T>(R_WC, R_CW);

        T v0[3], v_next[3]; // in the contact local coordinate
        matmul<3, 3, 1, T>(R_WC, v0_rel, v0);
        matmul<3, 3, 1, T>(R_WC, v_rel, v_next);

        T C_Hess[9], C_Grad[3]; // hess and grad in the contact local coordinate
        compute_contact_grad_and_hess(phi0, dt, stiffness, damping, friction_mu, v0, v_next, C_Hess, C_Grad);
        
        
        // hess and grad in the world local coordinate
        T W_Hess[9];
        T W_Grad[3];
        T tmp[9];
        matmul<3, 3, 1, T>(R_CW, C_Grad, W_Grad); // J^T yamma
        matmul<3, 3, 3, T>(R_CW, C_Hess, tmp);
        matmul<3, 3, 3, T>(tmp, R_WC, W_Hess); // J^T G J

        if constexpr(!JACOBI) {
            uint32_t i, j, k;
            get_color_coordinates(base[0], base[1], base[2], g_color_mask, i, j, k);

            T val[12]; // buffer for both W_Hess & W_Grad
            T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];
            #pragma unroll
            for (int ii = 0; ii < 9; ++ii) val[ii] = mass * weight * weight * W_Hess[ii];
            for (int ii = 9; ii < 12; ++ii) val[ii] = mass * weight * W_Grad[ii - 9];

            for (int iter = 1; iter <= mark; iter <<= 1) {
                T tmp[12]; 
                #pragma unroll
                for (int ii = 0; ii < 12; ++ii) tmp[ii] = __shfl_down_sync(0xFFFFFFFF, val[ii], iter);
                if (interval >= iter) {
                    #pragma unroll
                    for (int ii = 0; ii < 12; ++ii) val[ii] += tmp[ii];
                }
            }

            if (boundary) {
                const uint32_t target_cell_index = cell_index(base[0] + i, base[1] + j, base[2] + k);
                #pragma unroll
                for (int ii = 0; ii < 9; ++ii) atomicAdd(&(g_Hess[target_cell_index * 9 + ii]), val[ii]);
                #pragma unroll
                for (int ii = 0; ii < 3; ++ii) atomicAdd(&(g_Grad[target_cell_index * 3 + ii]), val[ii + 9]);
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        T val[12]; // buffer for both W_Hess & W_Grad
                            T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];
                            #pragma unroll
                            for (int ii = 0; ii < 9; ++ii) val[ii] = mass * weight * weight * W_Hess[ii];
                            for (int ii = 9; ii < 12; ++ii) val[ii] = mass * weight * W_Grad[ii - 9];

                            for (int iter = 1; iter <= mark; iter <<= 1) {
                                T tmp[12]; 
                                #pragma unroll
                                for (int ii = 0; ii < 12; ++ii) tmp[ii] = __shfl_down_sync(0xFFFFFFFF, val[ii], iter);
                                if (interval >= iter) {
                                    #pragma unroll
                                    for (int ii = 0; ii < 12; ++ii) val[ii] += tmp[ii];
                                }
                            }

                            if (boundary) {
                                const uint32_t target_cell_index = cell_index(base[0] + i, base[1] + j, base[2] + k);
                                #pragma unroll
                                for (int ii = 0; ii < 9; ++ii) atomicAdd(&(g_Hess[target_cell_index * 9 + ii]), val[ii]);
                                #pragma unroll
                                for (int ii = 0; ii < 3; ++ii) atomicAdd(&(g_Grad[target_cell_index * 3 + ii]), val[ii + 9]);
                            }
                    }
                }
            }
        }
    }
}

template<typename T, bool JACOBI>
__global__ void update_grid_contact_coordinate_descent_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    const T* g_v_star,
    T* g_Hess,
    T* g_Grad,
    T* g_momentum,
    T* g_Dir,
    T* g_alpha,
    T* g_E0,
    T* g_E1,
    T* norm_dir,
    uint32_t* total_grid_DoFs,
    const uint32_t g_color_mask,
    const T jacobi_relax_coeff) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(cell_idx);
        if (g_masses[cell_idx] > T(0.) && 
            (get_color_mask(xyz.x, xyz.y, xyz.z) == g_color_mask || JACOBI) &&
            (norm<9>(&g_Hess[cell_idx * 9]) > 1e-7 || norm<3>(&g_Grad[cell_idx * 3]) > 1e-7)) {
            T* g_vel = &g_momentum[cell_idx * 3];
            T mass = g_masses[cell_idx];
            T* local_Hess = &g_Hess[cell_idx * 9];
            T* local_Grad = &g_Grad[cell_idx * 3];
            T* local_Dir = &g_Dir[cell_idx * 3];
            local_Hess[0] -= mass;
            local_Hess[4] -= mass;
            local_Hess[8] -= mass;
            local_Grad[0] -= mass * (g_vel[0] - g_v_star[cell_idx * 3 + 0]);
            local_Grad[1] -= mass * (g_vel[1] - g_v_star[cell_idx * 3 + 1]);
            local_Grad[2] -= mass * (g_vel[2] - g_v_star[cell_idx * 3 + 2]);
            T Hess_Inv[9];
            inverse3(local_Hess, Hess_Inv);
            matmul<3, 3, 1, T>(Hess_Inv, local_Grad, local_Dir);

            // NOTE(changyu): enable it to test gradient descent
            // for (int i = 0; i < 3; ++i) local_Dir[i] = -local_Grad[i];

            // stop criterion
            atomicAdd(norm_dir, norm_sqr<3>(local_Dir));
            atomicAdd(total_grid_DoFs, 1U);

            // NOTE(changyu): enable it to add relaxation factor
            if (JACOBI) {
                for (int i = 0; i < 3; ++i) local_Dir[i] *= jacobi_relax_coeff;
            }

            g_alpha[cell_idx] = T(1.0);
            g_E0[cell_idx] = T(0.0);
            g_E1[cell_idx] = T(0.0);
        }
    }
}

template<typename T, int BLOCK_DIM, bool JACOBI, bool SOLVE_DF_DDF>
__global__ void grid_to_particle_vdb_line_search_kernel(const size_t n_particles,
    const T* contact_pos,
    const T* contact_vel,
    const T* velocities,
    const T* volumes,
    const uint32_t* contact_mpm_id,
    const T* contact_dist,
    const T* contact_normal,
    const T* contact_rigid_v,
    const T* g_velocities,
    const T* g_Dir,
    const T* g_alpha,
    T* g_E0,
    T* g_E1,
    T* g_dE1,
    T* g_d2E1,
    const T dt,
    const T friction_mu,
    const T stiffness,
    const T damping,
    const uint32_t g_color_mask,
    const bool eval_E0,
    const bool global_line_search,
    const T global_alpha) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // In [Fei et.al 2021],
    // we spill the B-spline weights (nine floats for each thread) by storing them into the shared memory
    // (instead of registers), although none of them is shared between threads. 
    // This choice increases performance, particularly when the number of threads is large (§6.2.5).
    __shared__ T weights[BLOCK_DIM][3][3];

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(contact_pos[idx * 3 + 0] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 1] * config::G_DX_INV<T> - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 2] * config::G_DX_INV<T> - T(0.5))
        };
        T fx[3] = {
            contact_pos[idx * 3 + 0] * config::G_DX_INV<T> - static_cast<T>(base[0]),
            contact_pos[idx * 3 + 1] * config::G_DX_INV<T> - static_cast<T>(base[1]),
            contact_pos[idx * 3 + 2] * config::G_DX_INV<T> - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        T old_v[3] = {0, 0, 0};
        T new_v[3] = {0, 0, 0};

        uint32_t ii, jj, kk;
        get_color_coordinates(base[0], base[1], base[2], g_color_mask, ii, jj, kk);
        const uint32_t color_index = cell_index(base[0] + ii, base[1] + jj, base[2] + kk);
        if (g_alpha[color_index] < 0. && !JACOBI) { // NOTE (changyu): use g_alpha[color_index] < 0 to indicate this DoF is solved
            return;
        }

        T global_Dir[3] = {0, 0, 0};

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    const uint32_t target_cell_index = cell_index(base[0] + i, base[1] + j, base[2] + k);
                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];

                    if constexpr (JACOBI) {
                        const T* g_v = &g_velocities[target_cell_index * 3];
                        const T* g_D = &g_Dir[target_cell_index * 3];
                        const T alpha = global_line_search ? global_alpha : g_alpha[target_cell_index];
                        old_v[0] += weight * g_v[0];
                        old_v[1] += weight * g_v[1];
                        old_v[2] += weight * g_v[2];
                        new_v[0] += weight * (g_v[0] - alpha * g_D[0]);
                        new_v[1] += weight * (g_v[1] - alpha * g_D[1]);
                        new_v[2] += weight * (g_v[2] - alpha * g_D[2]);
                        global_Dir[0] += weight * g_D[0];
                        global_Dir[1] += weight * g_D[1];
                        global_Dir[2] += weight * g_D[2];
                    } else {
                        if (get_color_mask(base[0] + i, base[1] + j, base[2] + k) == g_color_mask) {
                            const T* g_v = &g_velocities[target_cell_index * 3];
                            const T* g_D = &g_Dir[target_cell_index * 3];
                            const T alpha = g_alpha[target_cell_index];
                            old_v[0] += weight * g_v[0];
                            old_v[1] += weight * g_v[1];
                            old_v[2] += weight * g_v[2];
                            new_v[0] += weight * (g_v[0] - alpha * g_D[0]);
                            new_v[1] += weight * (g_v[1] - alpha * g_D[1]);
                            new_v[2] += weight * (g_v[2] - alpha * g_D[2]);
                        } else {
                            const T* g_v = &g_velocities[target_cell_index * 3];
                            old_v[0] += weight * g_v[0];
                            old_v[1] += weight * g_v[1];
                            old_v[2] += weight * g_v[2];
                            new_v[0] += weight * g_v[0];
                            new_v[1] += weight * g_v[1];
                            new_v[2] += weight * g_v[2];
                        }
                    }
                }
            }
        }

        const T mass = volumes[contact_mpm_id[idx]] * config::DENSITY<T>;
        const T* particle_v0 = &velocities[contact_mpm_id[idx] * 3];

        T nhat_W[3] = {-contact_normal[idx * 3 + 0], -contact_normal[idx * 3 + 1], -contact_normal[idx * 3 + 2]};
        T phi0 = -contact_dist[idx];

        T v0_rel[3] = {
            particle_v0[0] - contact_rigid_v[idx * 3 + 0],
            particle_v0[1] - contact_rigid_v[idx * 3 + 1],
            particle_v0[2] - contact_rigid_v[idx * 3 + 2]
        };
        T v_old_rel[3] = {
            old_v[0] - contact_rigid_v[idx * 3 + 0],
            old_v[1] - contact_rigid_v[idx * 3 + 1],
            old_v[2] - contact_rigid_v[idx * 3 + 2]
        };

        T v_new_rel[3] = {
            new_v[0] - contact_rigid_v[idx * 3 + 0],
            new_v[1] - contact_rigid_v[idx * 3 + 1],
            new_v[2] - contact_rigid_v[idx * 3 + 2]
        };

        constexpr int kZAxis = 2;
        T R_WC[9], R_CW[9]; // for each contact pair, Ji = R_CWp * wip
        make_from_one_unit_vector(nhat_W, kZAxis, R_WC);
        transpose<3, 3, T>(R_WC, R_CW);

        T v0[3], old_v_local[3], new_v_local[3]; // in the contact local coordinate
        matmul<3, 3, 1, T>(R_WC, v0_rel, v0);
        matmul<3, 3, 1, T>(R_WC, v_old_rel, old_v_local);
        matmul<3, 3, 1, T>(R_WC, v_new_rel, new_v_local);
        T v_hat = min(phi0 / dt, T(1.) / damping); // Eq. 

        auto l = [&](const T* v) {
            const T yn0 = max(stiffness * dt * phi0 * (T(1.) - damping * v0[kZAxis]), T(0.));
            const T lt = friction_mu * yn0 * (sqrt(v[0] * v[0] + v[1] * v[1] + config::epsv<T> * config::epsv<T>) - config::epsv<T>); // Eq. 33
            const T f0 = stiffness * phi0;
            const T vn = min(v_hat, v[kZAxis]); // Eq. 16
            const T ln_a = stiffness * damping * dt * dt;
            const T ln_b = -(stiffness * dt * (dt + damping * phi0));
            const T ln_c = stiffness * dt * phi0;
            const T ln = -(T(1. / 3.) * ln_a * vn * vn * vn + T(1. / 2.) * ln_b * vn * vn + ln_c * vn); // Eq.7
            return lt + ln;
        };

        T weight = weights[threadIdx.x][ii][0] * weights[threadIdx.x][jj][1] * weights[threadIdx.x][kk][2];
        if constexpr(SOLVE_DF_DDF) {
            if (JACOBI && global_line_search && !eval_E0) {
            } else {
                printf("SOLVE_DF_DDF must be JACOBI && global_line_search && !eval_E0!!!!!!!!!!!!!!!!!\n");
            }
        }
        if (global_line_search) {
            atomicAdd(g_E1, mass * l(new_v_local));
            if constexpr (SOLVE_DF_DDF) {
                T C_Hess[9], C_Grad[3]; // hess and grad in the contact local coordinate
                compute_contact_grad_and_hess(phi0, dt, stiffness, damping, friction_mu, v0, new_v_local, C_Hess, C_Grad);
                T R_WC_g_D[3];
                matmul<3, 3, 1, T>(R_WC, global_Dir, R_WC_g_D);
                atomicAdd(g_dE1, mass * dot<3>(C_Grad, R_WC_g_D));
                T tmp[3];
                matmul<1, 3, 3, T>(R_WC_g_D, C_Hess, tmp);
                atomicAdd(g_d2E1, mass * dot<3>(tmp, R_WC_g_D));
            }
        } else {
            if (JACOBI) {
                printf("ERROR, JACOBI CANNOT USE LOCAL LINE-SEARCH!!!!!!!!!!!!!!!!!!!!!!!\n");
            }
            atomicAdd(&g_E1[color_index], mass * l(new_v_local));
        }
        if (eval_E0) {
            if (global_line_search) {
                atomicAdd(g_E0, mass * l(old_v_local));
            } else {
                if (JACOBI) {
                    printf("ERROR, JACOBI CANNOT USE LOCAL LINE-SEARCH!!!!!!!!!!!!!!!!!!!!!!!\n");
                }
                atomicAdd(&g_E0[color_index], mass * l(old_v_local));
            }
        }
    }
}

template<typename T, bool JACOBI>
__global__ void update_grid_contact_alpha_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    const T* g_v_star,
    T* g_momentum,
    const T* g_Dir,
    T* g_alpha,
    const T* g_E0,
    T* g_E1,
    uint32_t* solved_grid_DoFs,
    const uint32_t g_color_mask,
    const bool enable_line_search) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(cell_idx);
        if (g_masses[cell_idx] > T(0.) && 
            (get_color_mask(xyz.x, xyz.y, xyz.z) == g_color_mask || JACOBI) &&
            g_alpha[cell_idx] > 0.) {
            T* g_vel = &g_momentum[cell_idx * 3];
            const T* v_star = &g_v_star[cell_idx * 3];
            const T mass = g_masses[cell_idx];
            const T alpha = g_alpha[cell_idx];
            const T* Dir = &g_Dir[cell_idx * 3];
            T old_v_rel[3] = {
                g_vel[0] - v_star[0],
                g_vel[1] - v_star[1],
                g_vel[2] - v_star[2]
            };
            T new_v_rel[3] = {
                g_vel[0] - alpha * Dir[0] - v_star[0],
                g_vel[1] - alpha * Dir[1] - v_star[1],
                g_vel[2] - alpha * Dir[2] - v_star[2]
            };
            T E0 = g_E0[cell_idx] + T(0.5) * mass * norm_sqr<3>(old_v_rel);
            T E1 = g_E1[cell_idx] + T(0.5) * mass * norm_sqr<3>(new_v_rel);
            if (E1 <= E0 || !enable_line_search) {
                g_vel[0] -= alpha * Dir[0];
                g_vel[1] -= alpha * Dir[1];
                g_vel[2] -= alpha * Dir[2];
                g_alpha[cell_idx] = T(-1.);
                atomicAdd(solved_grid_DoFs, 1U);
            } else {
                g_alpha[cell_idx] *= T(0.5);
                g_E1[cell_idx] = T(0.);
                if (g_alpha[cell_idx] < 1e-4) {
                    printf("Tiny Alpha!!!!!!!!!!! color=%u idx=%u E0=%.10f E1=%.10f\n", g_color_mask, cell_idx, E0, E1);
                    g_vel[0] -= alpha * Dir[0];
                    g_vel[1] -= alpha * Dir[1];
                    g_vel[2] -= alpha * Dir[2];
                    g_alpha[cell_idx] = T(-1.);
                    atomicAdd(solved_grid_DoFs, 1U);
                }
            }
        }
    }
}

template<typename T, bool JACOBI, bool SOLVE_DF_DDF>
__global__ void update_global_energy_grid_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    const T* g_v_star,
    T* g_momentum,
    const T* g_Dir,
    T* global_E0,
    T* global_E1,
    T* global_dE1,
    T* global_d2E1,
    const uint32_t g_color_mask,
    const T global_alpha,
    const bool eval_E0) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(cell_idx);
        if (g_masses[cell_idx] > T(0.) && 
            (get_color_mask(xyz.x, xyz.y, xyz.z) == g_color_mask || JACOBI)) {
            T* g_vel = &g_momentum[cell_idx * 3];
            const T* v_star = &g_v_star[cell_idx * 3];
            const T mass = g_masses[cell_idx];
            const T* Dir = &g_Dir[cell_idx * 3];
            T old_v_rel[3] = {
                g_vel[0] - v_star[0],
                g_vel[1] - v_star[1],
                g_vel[2] - v_star[2]
            };
            T new_v_rel[3] = {
                g_vel[0] - global_alpha * Dir[0] - v_star[0],
                g_vel[1] - global_alpha * Dir[1] - v_star[1],
                g_vel[2] - global_alpha * Dir[2] - v_star[2]
            };

            if constexpr(SOLVE_DF_DDF) {
                if (JACOBI && !eval_E0) {
                } else {
                    printf("SOLVE_DF_DDF must be JACOBI && !eval_E0!!!!!!!!!!!!!!!!!\n");
                }
            }
            if (eval_E0) {
                atomicAdd(global_E0, T(0.5) * mass * norm_sqr<3>(old_v_rel));
            }
            atomicAdd(global_E1, T(0.5) * mass * norm_sqr<3>(new_v_rel));
            if constexpr(SOLVE_DF_DDF) {
                atomicAdd(global_dE1, -mass * dot<3>(new_v_rel, Dir));
                atomicAdd(global_d2E1, mass * norm_sqr<3>(Dir));
            }
        }
    }
}

template<typename T, bool JACOBI>
__global__ void apply_global_line_search_grid_kernel(
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    T* g_momentum,
    const T* g_Dir,
    const uint32_t g_color_mask,
    const T global_alpha) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (config::G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (config::G_BLOCK_BITS * 3)) | (idx & config::G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(cell_idx);
        if (g_masses[cell_idx] > T(0.) && 
            (get_color_mask(xyz.x, xyz.y, xyz.z) == g_color_mask || JACOBI)) {
            T* g_vel = &g_momentum[cell_idx * 3];
            const T* Dir = &g_Dir[cell_idx * 3];
            g_vel[0] -= global_alpha * Dir[0];
            g_vel[1] -= global_alpha * Dir[1];
            g_vel[2] -= global_alpha * Dir[2];
        }
    }
}

template<typename T>
__global__ void apply_contact_impulse_to_rigid_bodies(
    const size_t n_contacts,
    const T* contact_pos,
    const T* contact_vel0,
    const T* contact_vel,
    const T* volumes,
    const uint32_t* contact_mpm_id,
    const uint32_t* contact_rigid_id,
    const T* contact_rigid_p_WB,
    T* F_Bq_W_tau,
    T* F_Bq_W_f) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_contacts) {
        T dv[3] = {
            contact_vel[idx * 3 + 0] - contact_vel0[idx * 3 + 0],
            contact_vel[idx * 3 + 1] - contact_vel0[idx * 3 + 1],
            contact_vel[idx * 3 + 2] - contact_vel0[idx * 3 + 2]
        };
        T m = volumes[contact_mpm_id[idx]] * config::DENSITY<T>;
        // We negate the sign of the grid node's momentum change to get
        //  the impulse applied to the rigid body at the grid node.
        T l_WN_W[3] = {m * -dv[0], m * -dv[1], m * -dv[2]};
        const T* p_WN = &contact_pos[idx * 3];
        const T* p_WB = &contact_rigid_p_WB[idx * 3];
        const T p_BN_W[3] = {
            p_WN[0] - p_WB[0],
            p_WN[1] - p_WB[1],
            p_WN[2] - p_WB[2]
        };
        // The angular impulse applied to the rigid body at the grid node.
        T h_WNBo_W[3];
        cross_product3(p_BN_W, l_WN_W, h_WNBo_W);

        // Use `F_Bq_W` to store the spatial impulse applied to the body
        //  at its origin, expressed in the world frame.
        #pragma unroll
        for (int i = 0 ; i < 3; ++i) {
            atomicAdd(&F_Bq_W_tau[contact_rigid_id[idx] * 3 + i], h_WNBo_W[i]);
            atomicAdd(&F_Bq_W_f[contact_rigid_id[idx] * 3 + i],   l_WN_W[i]);
        }
    }
}

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake