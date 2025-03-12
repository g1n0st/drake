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
__global__ void initialize_particle_state_kernel(
    const size_t n_particles,
    T *deformation_gradients) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        T *F = &deformation_gradients[idx * 9];
        F[0] = T(1.);
        F[1] = T(0.);
        F[2] = T(0.);

        F[3] = T(0.);
        F[4] = T(1.);
        F[5] = T(0.);

        F[6] = T(0.);
        F[7] = T(0.);
        F[8] = T(1.);
    }
}

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
        // NOTE (changyu): hard-code thickness as 1mm.
        // The parameter shouldn't vary with grid dx.
        T volume_4 = norm<3>(D0xD1) / T(8.) * T(0.001);

        volumes[idx] += volume_4;
        atomicAdd(&volumes[v0], volume_4);
        atomicAdd(&volumes[v1], volume_4);
        atomicAdd(&volumes[v2], volume_4);
    }
}

template<typename T>
__device__ __host__
inline void fixed_corotated_PK1_2D(const T* F, T* dphi_dF, const T mu, const T lambda) {
    T U[4], sig[4], V[4];
    svd2x2(F, U, sig, V);
    T R[4];
    matmulT<2, 2, 2, T>(U, V, R);
    T J = determinant2(F);
    T Finv[4];
    inverse2(F, Finv);
    dphi_dF[0] = T(2.) * mu * (F[0] - R[0]) + lambda * (J - T(1.)) * J * Finv[0];
    dphi_dF[1] = T(2.) * mu * (F[1] - R[1]) + lambda * (J - T(1.)) * J * Finv[2];
    dphi_dF[2] = T(2.) * mu * (F[2] - R[2]) + lambda * (J - T(1.)) * J * Finv[1];
    dphi_dF[3] = T(2.) * mu * (F[3] - R[3]) + lambda * (J - T(1.)) * J * Finv[3];
}

template<typename T>
__device__ __host__
inline void compute_dphi_dF(const T* F, T* dphi_dF, const T E, const T nu) {
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
    const T mu = E / (T(2.) * (T(1.) + nu));
	const T lambda = E * nu / ((T(1.) + nu) * (T(1.) - T(2.) * nu));
    fixed_corotated_PK1_2D(R_hat, dphi_dF_2x2, mu, lambda);
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
    // TM A = TM::Zero();
    // A(0, 0) = gp * sqr(s.R(0, 2));
    // A(0, 1) = gp * s.R(0, 2) * s.R(1, 2);
    // A(0, 2) = gp * s.R(2, 2) * s.R(0, 2);
    // A(1, 1) = gp * sqr(s.R(1, 2));
    // A(1, 2) = gp * s.R(2, 2) * s.R(1, 2);
    // A(2, 2) = fp * s.R(2, 2);
    // A(1, 0) = A(0, 1);
    // A(2, 0) = A(0, 2);
    // A(2, 1) = A(1, 2);
    A[0] = gp * R[2] * R[2];
    A[1] = gp * R[2] * R[5];
    A[2] = gp * R[8] * R[2];
    A[4] = gp * R[5] * R[5];
    A[5] = gp * R[8] * R[5];
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
    const T dt,
    const T E, 
    const T nu) {
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
        compute_dphi_dF(ctF, VP_local, E, nu);
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

template<typename T>
__global__ void calc_particle_state_and_force_kernel(
    const size_t n_particles,
    const T* volumes,
    const T* affine_matrices,
    T* deformation_gradients,
    T* taus,
    const T dt,
    const T E, 
    const T nu,
    const T yield_stress,
    const bool plasticity,
    const bool linear_corotated) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        T* F = &deformation_gradients[idx * 9];
        const T* C = &affine_matrices[idx * 9];
        
        float new_F[9];
        new_F[0] = (T(1.) + dt * C[0]) * F[0] + dt * C[1] * F[3] + dt * C[2] * F[6];
        new_F[1] = (T(1.) + dt * C[0]) * F[1] + dt * C[1] * F[4] + dt * C[2] * F[7];
        new_F[2] = (T(1.) + dt * C[0]) * F[2] + dt * C[1] * F[5] + dt * C[2] * F[8];

        new_F[3] = dt * C[3] * F[0] + (T(1.) + dt * C[4]) * F[3] + dt * C[5] * F[6];
        new_F[4] = dt * C[3] * F[1] + (T(1.) + dt * C[4]) * F[4] + dt * C[5] * F[7];
        new_F[5] = dt * C[3] * F[2] + (T(1.) + dt * C[4]) * F[5] + dt * C[5] * F[8];

        new_F[6] = dt * C[6] * F[0] + dt * C[7] * F[3] + (T(1.) + dt * C[8]) * F[6];
        new_F[7] = dt * C[6] * F[1] + dt * C[7] * F[4] + (T(1.) + dt * C[8]) * F[7];
        new_F[8] = dt * C[6] * F[2] + dt * C[7] * F[5] + (T(1.) + dt * C[8]) * F[8];

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            F[i] = new_F[i];
        }

        // NOTE, TODO (changyu): currently svd only supports float, all set float here
        float Uf[9], sigmaf[9], Vf[9];
        ssvd3x3<float>(new_F, Uf, sigmaf, Vf);
         
        T U[9], sigma[9], V[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            U[i] = Uf[i];
            sigma[i] = sigmaf[i];
            V[i] = Vf[i];
        }

        T *stress = &taus[idx * 9];
        // Lame parameters
        const T mu = E / (T(2.) * (T(1.) + nu));
	    const T lambda = E * nu / ((T(1.) + nu) * (T(1.) - T(2.) * nu));
        if (linear_corotated) {
            // Matrix3<T> R0;
            // Matrix3<T> strain;
            // Matrix3<T> unused_S;
            // fem::internal::PolarDecompose<T>(F0, &R0, &unused_S);
            // const Matrix3<T> corotated_F = R0.transpose() * FE;
            // strain =
            //     0.5 * (corotated_F + corotated_F.transpose()) - Matrix3<T>::Identity();
            // T trace_strain = strain.trace();
            // return {R0, strain, trace_strain};
            T tmp[9];
            T R[9], S[9];
            matmulT<3, 3, 3, T>(U, V, R);
            matmul<3, 3, 3, T>(V, sigma, tmp);
            matmulT<3, 3, 3, T>(tmp, V, S);
            T Rt[9];
            transpose<3, 3, T>(R, Rt);
            T corotated_F[9];
            matmul<3, 3, 3, T>(Rt, F, corotated_F);
            T strain[9];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    strain[i * 3 + j] = T(0.5) * (corotated_F[i * 3 + j] + corotated_F[j * 3 + i]) - T(i == j ? 1. : 0.);
                }
            T trace_strain = strain[0 * 3 + 0] + strain[1 * 3 + 1] + strain[2 * 3 + 2];

            T R_strain[9];
            matmul<3, 3, 3, T>(R, strain, R_strain);

            // StrainData data = ComputeStrainData(F0, FE);
            // (*P) = 2.0 * this->mu() * data.R0 * data.strain + this->lambda() * data.trace_strain * data.R0;
            T P[9];
            for (int i = 0; i < 9; ++i) {
                P[i] = T(2.) * mu * R_strain[i] + lambda * trace_strain * R[i];
            }

            // Matrix3<T> P;
            // CalcFirstPiolaStress(F0, FE, &P);
            // (*tau) = P * FE.transpose();
            matmulT<3, 3, 3, T>(P, F, stress);

            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                stress[i] *= volumes[idx];
            }
        }
        else {
            if (plasticity) {
                T b_trial[3] = {
                    sigma[0] * sigma[0],
                    sigma[4] * sigma[4],
                    sigma[8] * sigma[8]
                };

                T epsilon[3] = {
                    log(sigma[0]),
                    log(sigma[4]),
                    log(sigma[8])
                };

                T trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2];
                T epsilon_hat[3] = {
                    epsilon[0] - (trace_epsilon / T(3.)),
                    epsilon[1] - (trace_epsilon / T(3.)),
                    epsilon[2] - (trace_epsilon / T(3.))
                };
                T s_trial[3] = {
                    T(2.) * mu * epsilon_hat[0],
                    T(2.) * mu * epsilon_hat[1],
                    T(2.) * mu * epsilon_hat[2],
                };
                T s_trial_norm = norm<3>(s_trial) + T(1e-8);
                T y = s_trial_norm - sqrt(T(2./3.)) * yield_stress;
                if (y > 0) {
                    T mu_hat = mu * (b_trial[0] + b_trial[1] + b_trial[2]) / T(3.);
                    T s_new_norm = s_trial_norm - y;
                    T s_new[3] = {
                        (s_new_norm / s_trial_norm) * s_trial[0],
                        (s_new_norm / s_trial_norm) * s_trial[1],
                        (s_new_norm / s_trial_norm) * s_trial[2]
                    };
                    T H[3] = {
                        s_new[0] / (T(2.) * mu) + trace_epsilon / T(3.),
                        s_new[1] / (T(2.) * mu) + trace_epsilon / T(3.),
                        s_new[2] / (T(2.) * mu) + trace_epsilon / T(3.)
                    };

                    sigma[0 * 3 + 0] = exp(H[0]);
                    sigma[1 * 3 + 1] = exp(H[1]);
                    sigma[2 * 3 + 2] = exp(H[2]);
                }

                T tmp[9];
                matmul<3, 3, 3, T>(U, sigma, tmp);
                matmulT<3, 3, 3, T>(tmp, V, F);
            }
            
            // TODO (changyu): many register used here. Most of them could be reused to optimize performance.
            T J = determinant3(F);
            T R[9];
            matmulT<3, 3, 3, T>(U, V, R);
            
            T *two_mu_F_minus_R = R;
            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                two_mu_F_minus_R[i] = T(2.) * mu * (F[i] - R[i]);
            }
            matmulT<3, 3, 3, T>(two_mu_F_minus_R, F, stress);
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                stress[i * 3 + i] += lambda * J * (J - T(1.));
            }

            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                stress[i] *= volumes[idx];
            }
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

template<typename T>
__device__ __host__
inline std::uint32_t cell_index(const GridConfig<T> &gconf, const uint32_t &xi, const uint32_t &yi, const uint32_t &zi) noexcept {
    // NOTE (changyu): using morton code ordering within grid block (lower_bit) seems nothing different
    uint32_t higher_bit = morton_code({xi >> gconf.BLOCK_BITS, yi >> gconf.BLOCK_BITS, zi >> gconf.BLOCK_BITS});
    uint32_t lower_bit = ((xi & gconf.G_BLOCK_MASK) << (gconf.G_BLOCK_BITS * 2)) | ((yi & gconf.G_BLOCK_MASK) << gconf.G_BLOCK_BITS) | (zi & gconf.G_BLOCK_MASK);
    return (higher_bit << (gconf.G_BLOCK_BITS * 3)) | lower_bit;
    // printf("%.3lf %.3lf %.3lf %u %u %u high=%u, low=%u, %u\n", x, y, z, xi, yi, zi, higher_bit, lower_bit, keys[idx]);
}

template<typename T>
__device__ __host__
inline uint3 inverse_cell_index(const GridConfig<T> &gconf, const std::uint32_t &index) noexcept {
    // Extract higher_bit and lower_bit
    uint32_t higher_bit = index >> (gconf.G_BLOCK_BITS * 3);
    uint32_t lower_bit = index & ((1 << (gconf.G_BLOCK_BITS * 3)) - 1);

    // Extract xi, yi, zi from lower_bit
    uint32_t lower_xi = (lower_bit >> (gconf.G_BLOCK_BITS * 2)) & gconf.G_BLOCK_MASK;
    uint32_t lower_yi = (lower_bit >> gconf.G_BLOCK_BITS) & gconf.G_BLOCK_MASK;
    uint32_t lower_zi = lower_bit & gconf.G_BLOCK_MASK;

    // Extract xi, yi, zi from higher_bit using inverse Morton code
    uint3 higher_xyz = inverse_morton_code(higher_bit);

    // Combine higher and lower bits to get original xi, yi, zi
    uint32_t xi = (higher_xyz.x << gconf.BLOCK_BITS) | lower_xi;
    uint32_t yi = (higher_xyz.y << gconf.BLOCK_BITS) | lower_yi;
    uint32_t zi = (higher_xyz.z << gconf.BLOCK_BITS) | lower_zi;

    return {xi, yi, zi};
}

template<typename T>
__global__ void compute_base_cell_node_index_kernel(const GridConfig<T> gconf, const size_t n_particles, const T* positions, uint32_t* keys, uint32_t* ids) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        T x = positions[idx * 3 + 0];
        T y = positions[idx * 3 + 1];
        T z = positions[idx * 3 + 2];
        uint32_t xi = static_cast<uint32_t>(x * gconf.G_DX_INV - T(0.5));
        uint32_t yi = static_cast<uint32_t>(y * gconf.G_DX_INV - T(0.5));
        uint32_t zi = static_cast<uint32_t>(z * gconf.G_DX_INV - T(0.5));
        keys[idx] = cell_index(gconf, xi, yi, zi);
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

template<typename T, int BLOCK_DIM>
__global__ void particle_to_grid_kernel(const GridConfig<T> gconf, 
    const size_t n_particles,
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
            static_cast<uint32_t>(positions[idx * 3 + 0] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 1] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 2] * gconf.G_DX_INV - T(0.5))
        };
        T fx[3] = {
            positions[idx * 3 + 0] * gconf.G_DX_INV - static_cast<T>(base[0]),
            positions[idx * 3 + 1] * gconf.G_DX_INV - static_cast<T>(base[1]),
            positions[idx * 3 + 2] * gconf.G_DX_INV - static_cast<T>(base[2])
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
            B[i] = (-dt * gconf.G_DX_INV) * stress[i] + C[i] * mass;
        }

        T val[4];

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    T xi_minus_xp[3] = {
                        (i - fx[0]) * gconf.G_DX,
                        (j - fx[1]) * gconf.G_DX,
                        (k - fx[2]) * gconf.G_DX
                    };

                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];

                    val[0] = mass * weight;
                    val[1] = vel[0] * val[0];
                    val[2] = vel[1] * val[0];
                    val[3] = vel[2] * val[0];
                    // apply gravity
                    val[config::GRAVITY_AXIS + 1] += val[0] * config::GRAVITY<T> * dt;

                    val[1] += (B[0] * xi_minus_xp[0] + B[1] * xi_minus_xp[1] + B[2] * xi_minus_xp[2]) * weight;
                    val[2] += (B[3] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] + B[5] * xi_minus_xp[2]) * weight;
                    val[3] += (B[6] * xi_minus_xp[0] + B[7] * xi_minus_xp[1] + B[8] * xi_minus_xp[2]) * weight;
                    const T* force = &forces[idx * 3];
                    val[1] += force[0] * dt * weight;
                    val[2] += force[1] * dt * weight;
                    val[3] += force[2] * dt * weight;

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
                        const uint32_t target_cell_index = cell_index(gconf, base[0] + i, base[1] + j, base[2] + k);
                        const uint32_t target_grid_index = target_cell_index >> (gconf.G_BLOCK_BITS * 3);
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
    const int G_GRID_VOLUME,
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

    if (idx < G_GRID_VOLUME && g_touched_flags[idx]) {
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
    const GridConfig<T> gconf,
    const uint32_t touched_cells_cnt,
    const uint32_t* g_touched_ids,
    uint32_t* g_touched_flags,
    T* g_masses,
    T* g_momentum) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
        g_touched_flags[block_idx] = 0;
        g_masses[cell_idx] = 0;
        g_momentum[cell_idx * 3 + 0] = 0;
        g_momentum[cell_idx * 3 + 1] = 0;
        g_momentum[cell_idx * 3 + 2] = 0;
    }
}

template<typename T>
__global__ void clean_grid_contact_kernel(
    const GridConfig<T> gconf,
    const uint32_t touched_cells_cnt,
    const uint32_t* g_touched_ids,
    T* g_Hess,
    T* g_Grad,
    T* g_Dir) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
        g_Grad[cell_idx * 3 + 0] = 0;
        g_Grad[cell_idx * 3 + 1] = 0;
        g_Grad[cell_idx * 3 + 2] = 0;
        g_Dir[cell_idx * 3 + 0] = 0;
        g_Dir[cell_idx * 3 + 1] = 0;
        g_Dir[cell_idx * 3 + 2] = 0;
        #pragma unroll
        for (int i = 0; i < 9; ++i) g_Hess[cell_idx * 9 + i] = 0;
    }
}

template<typename T, int MPM_BOUNDARY_CONDITION=-1, bool ENFORCE_BC_ONLY=false>
__global__ void update_grid_kernel(
    const GridConfig<T> gconf,
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    T* g_masses,
    T* g_momentum,
    T* g_v_star,
    const T times_elapsed) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
        if (g_masses[cell_idx] > T(0.)) {
            T *g_vel = &g_momentum[cell_idx * 3];

            if constexpr (!ENFORCE_BC_ONLY) {
                // printf("m=%lf mv=(%lf %lf %lf)\n", g_masses[cell_idx], g_vel[0], g_vel[1], g_vel[2]);
                g_vel[0] /= g_masses[cell_idx];
                g_vel[1] /= g_masses[cell_idx];
                g_vel[2] /= g_masses[cell_idx];
            }
            else {

            // apply boundary condition
            const int boundary_condition = 3;
            uint3 xyz = inverse_cell_index(gconf, cell_idx);
            if (xyz.x < boundary_condition && g_vel[0] < 0) g_vel[0] = 0;
            if (xyz.x >= gconf.G_DOMAIN_SIZE - boundary_condition && g_vel[0] > 0) g_vel[0] = 0;
            if (xyz.y < boundary_condition && g_vel[1] < 0) g_vel[1] = 0;
            if (xyz.y >= gconf.G_DOMAIN_SIZE - boundary_condition && g_vel[1] > 0) g_vel[1] = 0;
            if (xyz.z < boundary_condition && g_vel[2] < 0) g_vel[2] = 0;
            if (xyz.z >= gconf.G_DOMAIN_SIZE - boundary_condition && g_vel[2] > 0) g_vel[2] = 0;

            {
                T pos[3] = {
                    (xyz.x + T(.5)) * gconf.G_DX,
                    (xyz.y + T(.5)) * gconf.G_DX,
                    (xyz.z + T(.5)) * gconf.G_DX
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

                // z-axis=0.5 used for particle (dual_arm/roll/) demos
                else if constexpr (MPM_BOUNDARY_CONDITION == 111) {
                    normal[0] = T(0.);
                    normal[1] = T(0.);
                    normal[2] = T(1.);
                    dist = pos[2] - T(0.5);
                    if (dist < 0) {
                        inside = true;
                        diff_vel[0] = -g_vel[0];
                        diff_vel[1] = -g_vel[1];
                        diff_vel[2] = -g_vel[2];
                        dotnv = dot<3>(diff_vel, normal);
                    }
                }

                // z-axis=0.005 used for cloth (dual_arm_folding) demos
                else if constexpr (MPM_BOUNDARY_CONDITION == 222) {
                    normal[0] = T(0.);
                    normal[1] = T(0.);
                    normal[2] = T(1.);
                    dist = pos[2] - T(0.02);
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

                // for allegro bagging demo
                else if constexpr (MPM_BOUNDARY_CONDITION == 114) {
                    constexpr T gripper_xy = 0.1 / 2.0;
                    constexpr T gripper_z = 0.04 / 2.0;
                    
                    constexpr T l_x = 0.34 - 0.03 - 0.01;
                    constexpr T h_x = 0.66 + 0.027 + 0.01;
                    constexpr T l_z = 0.39-2e-4 - 0.01;
                    constexpr T h_z = 0.41+2e-4 + 0.01;
                    
                    constexpr T initial_free_duration = 0.25;
                    constexpr T initial_loose_duration = 0.25;
                    constexpr T free_duration = 8.0;
                    constexpr T bagging_duration = 1.0 - initial_loose_duration;
                    constexpr T static_duration = 1.0;
                    constexpr T final_loose_duration = 0.4;
                    constexpr T bagging_v = 0.1;

                    T gll_up_p[3];
                    T glh_up_p[3];
                    T ghl_up_p[3];
                    T ghh_up_p[3];
                    T gll_down_p[3];
                    T glh_down_p[3];
                    T ghl_down_p[3];
                    T ghh_down_p[3];

                    T gll_up_v[3];
                    T glh_up_v[3];
                    T ghl_up_v[3];
                    T ghh_up_v[3];
                    T gll_down_v[3];
                    T glh_down_v[3];
                    T ghl_down_v[3];
                    T ghh_down_v[3];

                    const auto& assign = [](T *x, const T a, const T b, const T c) {
                        x[0] = a; 
                        x[1] = b; 
                        x[2] = c;
                    };

                    const auto& check = [&](const T *gx, const T* gv, const bool is_up) {
                        if (is_up) {
                            if (
                                pos[0] >= gx[0] - gripper_xy && pos[0] <= gx[0] + gripper_xy &&
                                pos[1] >= gx[1] - gripper_xy && pos[1] <= gx[1] + gripper_xy &&
                                pos[2] >= gx[2] - gripper_z
                            ) {
                                dist = gx[2] - gripper_z - pos[2];
                                normal[0] = T(0.);
                                normal[1] = T(0.);
                                normal[2] = T(-1.);
                                diff_vel[0] = gv[0] - g_vel[0];
                                diff_vel[1] = gv[1] - g_vel[1];
                                diff_vel[2] = gv[2] - g_vel[2];
                                dotnv = dot<3>(normal, diff_vel);
                                inside = true;
                                fixed = true;
                            }
                        } else {
                            if (
                                pos[0] >= gx[0] - gripper_xy && pos[0] <= gx[0] + gripper_xy &&
                                pos[1] >= gx[1] - gripper_xy && pos[1] <= gx[1] + gripper_xy &&
                                pos[2] <= gx[2] + gripper_z
                            ) {
                                dist = pos[2] - gx[2] - gripper_z;
                                normal[0] = T(0.);
                                normal[1] = T(0.);
                                normal[2] = T(1.);
                                diff_vel[0] = gv[0] - g_vel[0];
                                diff_vel[1] = gv[1] - g_vel[1];
                                diff_vel[2] = gv[2] - g_vel[2];
                                dotnv = dot<3>(normal, diff_vel);
                                inside = true;
                                fixed = true;
                            }
                        }
                    };

                    if (times_elapsed < initial_free_duration) {
                        assign(gll_up_p, l_x, l_x, h_z);
                        assign(glh_up_p, l_x, h_x, h_z);
                        assign(ghl_up_p, h_x, l_x,  h_z);
                        assign(ghh_up_p, h_x, h_x, h_z);
                        assign(gll_down_p, l_x, l_x, l_z);
                        assign(glh_down_p, l_x, h_x, l_z);
                        assign(ghl_down_p, h_x, l_x,  l_z);
                        assign(ghh_down_p, h_x, h_x, l_z);

                        assign(gll_up_v, 0, 0, 0);
                        assign(glh_up_v, 0, 0, 0);
                        assign(ghl_up_v, 0, 0, 0);
                        assign(ghh_up_v, 0, 0, 0);
                        assign(gll_down_v, 0, 0, 0);
                        assign(glh_down_v, 0, 0, 0);
                        assign(ghl_down_v, 0, 0, 0);
                        assign(ghh_down_v, 0, 0, 0);
                    } else if (times_elapsed < initial_free_duration + initial_loose_duration) {
                        double dt = times_elapsed - initial_free_duration;
                        assign(gll_up_p, l_x + dt * bagging_v, l_x + dt * bagging_v, h_z);
                        assign(glh_up_p, l_x + dt * bagging_v, h_x - dt * bagging_v, h_z);
                        assign(ghl_up_p, h_x - dt * bagging_v, l_x + dt * bagging_v,  h_z);
                        assign(ghh_up_p, h_x - dt * bagging_v, h_x - dt * bagging_v, h_z);
                        assign(gll_down_p, l_x + dt * bagging_v, l_x + dt * bagging_v, l_z);
                        assign(glh_down_p, l_x + dt * bagging_v, h_x - dt * bagging_v, l_z);
                        assign(ghl_down_p, h_x - dt * bagging_v, l_x + dt * bagging_v,  l_z);
                        assign(ghh_down_p, h_x - dt * bagging_v, h_x - dt * bagging_v, l_z);

                        assign(gll_up_v, + bagging_v, + bagging_v, 0);
                        assign(glh_up_v, + bagging_v, - bagging_v, 0);
                        assign(ghl_up_v, - bagging_v, + bagging_v, 0);
                        assign(ghh_up_v, - bagging_v, - bagging_v, 0);
                        assign(gll_down_v, + bagging_v, + bagging_v, 0);
                        assign(glh_down_v, + bagging_v, - bagging_v, 0);
                        assign(ghl_down_v, - bagging_v, + bagging_v, 0);
                        assign(ghh_down_v, - bagging_v, - bagging_v, 0);
                    } else if (times_elapsed < free_duration + initial_loose_duration + initial_free_duration) {
                        double dt = initial_loose_duration;
                        assign(gll_up_p, l_x + dt * bagging_v, l_x + dt * bagging_v, h_z);
                        assign(glh_up_p, l_x + dt * bagging_v, h_x - dt * bagging_v, h_z);
                        assign(ghl_up_p, h_x - dt * bagging_v, l_x + dt * bagging_v,  h_z);
                        assign(ghh_up_p, h_x - dt * bagging_v, h_x - dt * bagging_v, h_z);
                        assign(gll_down_p, l_x + dt * bagging_v, l_x + dt * bagging_v, l_z);
                        assign(glh_down_p, l_x + dt * bagging_v, h_x - dt * bagging_v, l_z);
                        assign(ghl_down_p, h_x - dt * bagging_v, l_x + dt * bagging_v,  l_z);
                        assign(ghh_down_p, h_x - dt * bagging_v, h_x - dt * bagging_v, l_z);
                    
                        assign(gll_up_v, 0, 0, 0);
                        assign(glh_up_v, 0, 0, 0);
                        assign(ghl_up_v, 0, 0, 0);
                        assign(ghh_up_v, 0, 0, 0);
                        assign(gll_down_v, 0, 0, 0);
                        assign(glh_down_v, 0, 0, 0);
                        assign(ghl_down_v, 0, 0, 0);
                        assign(ghh_down_v, 0, 0, 0);
                    } else if (times_elapsed < free_duration + bagging_duration + initial_loose_duration + initial_free_duration) {
                        double dt = (times_elapsed - free_duration - initial_free_duration);
                        assign(gll_up_p, l_x + dt * bagging_v, l_x + dt * bagging_v, h_z);
                        assign(glh_up_p, l_x + dt * bagging_v, h_x - dt * bagging_v, h_z);
                        assign(ghl_up_p, h_x - dt * bagging_v, l_x + dt * bagging_v,  h_z);
                        assign(ghh_up_p, h_x - dt * bagging_v, h_x - dt * bagging_v, h_z);
                        assign(gll_down_p, l_x + dt * bagging_v, l_x + dt * bagging_v, l_z);
                        assign(glh_down_p, l_x + dt * bagging_v, h_x - dt * bagging_v, l_z);
                        assign(ghl_down_p, h_x - dt * bagging_v, l_x + dt * bagging_v,  l_z);
                        assign(ghh_down_p, h_x - dt * bagging_v, h_x - dt * bagging_v, l_z);
                    
                        assign(gll_up_v, + bagging_v, + bagging_v, 0);
                        assign(glh_up_v, + bagging_v, - bagging_v, 0);
                        assign(ghl_up_v, - bagging_v, + bagging_v, 0);
                        assign(ghh_up_v, - bagging_v, - bagging_v, 0);
                        assign(gll_down_v, + bagging_v, + bagging_v, 0);
                        assign(glh_down_v, + bagging_v, - bagging_v, 0);
                        assign(ghl_down_v, - bagging_v, + bagging_v, 0);
                        assign(ghh_down_v, - bagging_v, - bagging_v, 0);
                    } else if (times_elapsed < free_duration + bagging_duration + initial_loose_duration + initial_free_duration + static_duration) {
                        double total_dur = initial_loose_duration + bagging_duration;
                        assign(gll_up_p, l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, h_z);
                        assign(glh_up_p, l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, h_z);
                        assign(ghl_up_p, h_x - total_dur * bagging_v, l_x + total_dur * bagging_v,  h_z);
                        assign(ghh_up_p, h_x - total_dur * bagging_v, h_x - total_dur * bagging_v, h_z);
                        assign(gll_down_p, l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, l_z);
                        assign(glh_down_p, l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, l_z);
                        assign(ghl_down_p, h_x - total_dur * bagging_v, l_x + total_dur * bagging_v,  l_z);
                        assign(ghh_down_p, h_x - total_dur * bagging_v, h_x - total_dur * bagging_v, l_z);
                    
                        assign(gll_up_v, 0, 0, 0);
                        assign(glh_up_v, 0, 0, 0);
                        assign(ghl_up_v, 0, 0, 0);
                        assign(ghh_up_v, 0, 0, 0);
                        assign(gll_down_v, 0, 0, 0);
                        assign(glh_down_v, 0, 0, 0);
                        assign(ghl_down_v, 0, 0, 0);
                        assign(ghh_down_v, 0, 0, 0);
                    } else {
                        double total_dur = initial_loose_duration + bagging_duration;
                        double dt = 100.0;
                        assign(gll_up_p, l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, h_z);
                        assign(glh_up_p, l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, h_z);
                        assign(ghl_up_p, h_x - (total_dur - dt) * bagging_v, l_x + (total_dur - dt) * bagging_v,  h_z + dt * bagging_v);
                        assign(ghh_up_p, h_x - (total_dur - dt) * bagging_v, h_x - (total_dur - dt) * bagging_v, h_z + dt * bagging_v);
                        assign(gll_down_p, l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, l_z);
                        assign(glh_down_p, l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, l_z);
                        assign(ghl_down_p, h_x - (total_dur - dt) * bagging_v, l_x + (total_dur - dt) * bagging_v,  l_z - dt * bagging_v);
                        assign(ghh_down_p, h_x - (total_dur - dt) * bagging_v, h_x - (total_dur - dt) * bagging_v, l_z - dt * bagging_v);

                        assign(gll_up_v, 0, 0, 0);
                        assign(glh_up_v, 0, 0, 0);
                        assign(ghl_up_v, bagging_v, -bagging_v, + bagging_v);
                        assign(ghh_up_v, bagging_v, +bagging_v, + bagging_v);
                        assign(gll_down_v, 0, 0, 0);
                        assign(glh_down_v, 0, 0, 0);
                        assign(ghl_down_v, bagging_v, -bagging_v, - bagging_v);
                        assign(ghh_down_v, bagging_v, +bagging_v, - bagging_v);
                    }

                    check(gll_up_p, gll_up_v, true);
                    check(glh_up_p, glh_up_v, true);
                    check(ghl_up_p, ghl_up_v, true);
                    check(ghh_up_p, ghh_up_v, true);
                    check(gll_down_p, gll_down_v, false);
                    check(glh_down_p, glh_down_v, false);
                    check(ghl_down_p, ghl_down_v, false);
                    check(ghh_down_p, ghh_down_v, false);
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
            }

            if constexpr (!ENFORCE_BC_ONLY) {
                g_v_star[cell_idx * 3 + 0] = g_vel[0];
                g_v_star[cell_idx * 3 + 1] = g_vel[1];
                g_v_star[cell_idx * 3 + 2] = g_vel[2];
            }
        }
    }
}

template<typename T, int BLOCK_DIM, bool CONTACT_TRANSFER>
__global__ void grid_to_particle_kernel(
    const GridConfig<T> gconf,
    const size_t n_particles,
    T* positions, 
    T* velocities,
    T* affine_matrices,
    const T* g_masses,
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
            static_cast<uint32_t>(positions[idx * 3 + 0] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 1] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(positions[idx * 3 + 2] * gconf.G_DX_INV - T(0.5))
        };
        T fx[3] = {
            positions[idx * 3 + 0] * gconf.G_DX_INV - static_cast<T>(base[0]),
            positions[idx * 3 + 1] * gconf.G_DX_INV - static_cast<T>(base[1]),
            positions[idx * 3 + 2] * gconf.G_DX_INV - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        T new_v[3];
        T new_C[9], new_CT[9];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            new_v[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            new_C[i] = 0;
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

                    const uint32_t target_cell_index = cell_index(gconf, base[0] + i, base[1] + j, base[2] + k);
                    const T* g_v = &g_momentum[target_cell_index * 3];

                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];

                    if constexpr (CONTACT_TRANSFER) {
                        new_v[0] += weight * g_v[0];
                        new_v[1] += weight * g_v[1];
                        new_v[2] += weight * g_v[2];
                    } else {
                        new_v[0] += weight * g_v[0];
                        new_v[1] += weight * g_v[1];
                        new_v[2] += weight * g_v[2];
                        // printf("weight=%lf, g_v=(%lf %lf %lf)\n", weight, g_v[0], g_v[1], g_v[2]);

                        // printf("i=%d j=%d k=%d\n", i, j, k);
                        // printf("weight=%.8lf\n", weight);
                        // printf("g_v=[%.8lf   %.8lf   %.8lf]\n", g_v[0], g_v[1], g_v[2]);
                        // printf("xip=[%.8lf   %.8lf   %.8lf]\n", xi_minus_xp[0], xi_minus_xp[1], xi_minus_xp[2]);
                        new_C[0] += 4 * gconf.G_DX_INV * weight * g_v[0] * xi_minus_xp[0];
                        new_C[1] += 4 * gconf.G_DX_INV * weight * g_v[0] * xi_minus_xp[1];
                        new_C[2] += 4 * gconf.G_DX_INV * weight * g_v[0] * xi_minus_xp[2];
                        new_C[3] += 4 * gconf.G_DX_INV * weight * g_v[1] * xi_minus_xp[0];
                        new_C[4] += 4 * gconf.G_DX_INV * weight * g_v[1] * xi_minus_xp[1];
                        new_C[5] += 4 * gconf.G_DX_INV * weight * g_v[1] * xi_minus_xp[2];
                        new_C[6] += 4 * gconf.G_DX_INV * weight * g_v[2] * xi_minus_xp[0];
                        new_C[7] += 4 * gconf.G_DX_INV * weight * g_v[2] * xi_minus_xp[1];
                        new_C[8] += 4 * gconf.G_DX_INV * weight * g_v[2] * xi_minus_xp[2];
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
            affine_matrices[idx * 9 + 0] = ((config::V<T> + T(1.)) * T(.5)) * new_C[0] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[0];
            affine_matrices[idx * 9 + 1] = ((config::V<T> + T(1.)) * T(.5)) * new_C[1] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[1];
            affine_matrices[idx * 9 + 2] = ((config::V<T> + T(1.)) * T(.5)) * new_C[2] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[2];
            affine_matrices[idx * 9 + 3] = ((config::V<T> + T(1.)) * T(.5)) * new_C[3] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[3];
            affine_matrices[idx * 9 + 4] = ((config::V<T> + T(1.)) * T(.5)) * new_C[4] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[4];
            affine_matrices[idx * 9 + 5] = ((config::V<T> + T(1.)) * T(.5)) * new_C[5] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[5];
            affine_matrices[idx * 9 + 6] = ((config::V<T> + T(1.)) * T(.5)) * new_C[6] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[6];
            affine_matrices[idx * 9 + 7] = ((config::V<T> + T(1.)) * T(.5)) * new_C[7] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[7];
            affine_matrices[idx * 9 + 8] = ((config::V<T> + T(1.)) * T(.5)) * new_C[8] + ((config::V<T> - T(1.)) * T(.5)) * new_CT[8];

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

// SAP model
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

    // NOTE (changyu): in math eqns from (https://arxiv.org/pdf/2312.03908) and in code,
    // ϕ differs by a sign.
    const T phi = phi0 - dt * v_next[kZAxis];
    // If (-ϕ0 - δt vn)+ or (1 − dvn)+ equals zero, no impulse should be applied
    if (T(1.) - damping * v_next[kZAxis] <= 0 || phi <= 0) { // Quick exits
        #pragma unroll
        for (int i = 0; i < 9; ++i) C_Hess[i] = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i) C_Grad[i] = 0;
    }
    else {
        // normal component (Compliant Contact)
        // fn(ϕ, vn) = k (−ϕ)+ (1 − dvn)+
        // γn(vn) = n(vn; ϕ0) = δt fn(ϕ0 + δt vn, vn)
        //        = δt k (-ϕ0 - δt vn)+ (1 − dvn)+
        const T yn = dt * stiffness * (phi0 - dt * v_next[kZAxis]) * (T(1.) - damping * v_next[kZAxis]); // Eq. 13

        // ∂²ln / ∂vn² = - δt ∂ fn / ∂vn
        //               = - δt ∂ fn(ϕ0 + δt vn, vn) / ∂vn
        //               = - δt k ∂ (-ϕ0 - δt vn)+ (1 − dvn)+ / ∂vn
        // when both (-ϕ0 - δt vn) > 0 and (1 − dvn) > 0 satisfied
        //               = - δt k ∂ (-ϕ0 - δt vn) (1 − dvn) / ∂vn
        //               = - δt k ∂ (-ϕ0 - δt vn + ϕ0 dvn + d δt vn²) / ∂vn
        //               = - δt k (- δt + ϕ0 d + 2 d δt vn)
        const T d2lndvn2 = - dt * stiffness * (-dt -phi0 * damping + T(2.) * damping * dt * v_next[kZAxis]); // Eq. 8

        // frictional component (Lagged Model)
        // For a physical model of compliance for which γn is only a function of vn

        // γn0 = δt fn(ϕ0, vn0) = δt k (−ϕ0)+ (1 − dvn0)+
        const T yn0 = max(stiffness * dt * phi0 * (T(1.) - damping * v0[kZAxis]), T(0.));

        const T ts_coeff = sqrt(v_next[0] * v_next[0] + v_next[1] * v_next[1] + config::epsv<T> * config::epsv<T>);
        const T ts_hat[2] = {v_next[0] / ts_coeff, v_next[1] / ts_coeff}; // Eq. 18

        // γt = -μ * γn0 * t̂_s 
        const T yt[2] = {
            -friction_mu * yn0 * ts_hat[0], 
            -friction_mu * yn0 * ts_hat[1]
        }; // Eq. 33

        T P_ts_hat[4];
        outer_product<2, T>(ts_hat, ts_hat, P_ts_hat);
        const T P_perp_ts_hat[4] = {
            T(1.) - P_ts_hat[0], -P_ts_hat[1],
            -P_ts_hat[2], T(1.) - P_ts_hat[3]
        };
        const T d2ltdvt2_coeff = friction_mu * yn0 / ts_coeff; // Eq. 33, ts_coeff = ts_soft_norm + epsv
        // ∂²lt / ∂vt² = μ * γn0 * (P⊥(t̂_s) / (||v_t||_s + ε_s))
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

        C_Grad[0] = -yt[0];
        C_Grad[1] = -yt[1];
        C_Grad[2] = -yn;
    }
}

template<typename T, int BLOCK_DIM>
__global__ void contact_particle_to_grid_kernel(
    const GridConfig<T> gconf,
    const size_t n_particles,
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
    const T damping) {
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
            static_cast<uint32_t>(contact_pos[idx * 3 + 0] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 1] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 2] * gconf.G_DX_INV - T(0.5))
        };
        T fx[3] = {
            contact_pos[idx * 3 + 0] * gconf.G_DX_INV - static_cast<T>(base[0]),
            contact_pos[idx * 3 + 1] * gconf.G_DX_INV - static_cast<T>(base[1]),
            contact_pos[idx * 3 + 2] * gconf.G_DX_INV - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        const T mass = volumes[contact_mpm_id[idx]] * config::DENSITY<T>;
        const T* particle_vn = &velocities[contact_mpm_id[idx] * 3];
        const T* particle_v = &contact_vel[idx * 3];

        T nhat_W[3] = {contact_normal[idx * 3 + 0], contact_normal[idx * 3 + 1], contact_normal[idx * 3 + 2]};
        T phi0 = -contact_dist[idx];

        T vn_rel_W[3] = {
            particle_vn[0] - contact_rigid_v[idx * 3 + 0],
            particle_vn[1] - contact_rigid_v[idx * 3 + 1],
            particle_vn[2] - contact_rigid_v[idx * 3 + 2]
        };
        T v_rel_W[3] = {
            particle_v[0] - contact_rigid_v[idx * 3 + 0],
            particle_v[1] - contact_rigid_v[idx * 3 + 1],
            particle_v[2] - contact_rigid_v[idx * 3 + 2]
        };

        constexpr int kZAxis = 2;
        T R_WC[9], R_CW[9]; // for each contact pair, Ji = R_CWp * wip
        make_from_one_unit_vector(nhat_W, kZAxis, R_WC);
        transpose<3, 3, T>(R_WC, R_CW);

        T vn_C[3], v_next_C[3]; // in the contact local coordinate
        matmul<3, 3, 1, T>(R_CW, vn_rel_W, vn_C);
        matmul<3, 3, 1, T>(R_CW, v_rel_W, v_next_C);

        T lc_Hess_C[9], lc_Grad_C[3]; // hess and grad in the contact local coordinate
        compute_contact_grad_and_hess(phi0, dt, stiffness, damping, friction_mu, vn_C, v_next_C, lc_Hess_C, lc_Grad_C);
        
        
        // hess and grad in the world local coordinate
        T lc_Hess_W[9];
        T lc_Grad_W[3];
        T tmp[9];

        // Grad/Hess for lc(vp(vi))
        matmul<3, 3, 1, T>(R_WC, lc_Grad_C, lc_Grad_W); // J^T Grad
        matmul<3, 3, 3, T>(R_WC, lc_Hess_C, tmp);
        matmul<3, 3, 3, T>(tmp, R_CW, lc_Hess_W); // J^T Hess J

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    T val[12]; // buffer for both W_Hess & W_Grad
                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];
                    #pragma unroll
                    for (int ii = 0; ii < 9; ++ii) val[ii] = weight * weight * lc_Hess_W[ii];
                    for (int ii = 9; ii < 12; ++ii) val[ii] = weight * lc_Grad_W[ii - 9];

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
                        const uint32_t target_cell_index = cell_index(gconf, base[0] + i, base[1] + j, base[2] + k);
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

template<typename T>
__global__ void grid_contact_3x3_parallel_solving_kernel(
    const GridConfig<T> gconf,
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    const T* g_v_star,
    T* g_Hess,
    T* g_Grad,
    T* g_momentum,
    T* g_Dir,
    T* norm_dir,
    T* norm_impulse) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(gconf, cell_idx);
        if (g_masses[cell_idx] > T(0.)) {
            T* g_vel = &g_momentum[cell_idx * 3];
            T mass = g_masses[cell_idx];
            T* local_Hess = &g_Hess[cell_idx * 9];
            T* local_Grad = &g_Grad[cell_idx * 3];
            T* local_Dir = &g_Dir[cell_idx * 3];
            T sqrt_mass = sqrt(mass);

            // M + J^T * G * J
            local_Hess[0] += mass;
            local_Hess[4] += mass;
            local_Hess[8] += mass;

            // TODO(xuchenhan-tri): We only need to compute the norm of the scaled impulse and momentum
            //  at the first iteration.
            // We scale the impulse and momentum by the square root of the mass following the original SAP paper.
            T scaled_impulse[3] = {
                -local_Grad[0] / sqrt_mass,
                -local_Grad[1] / sqrt_mass,
                -local_Grad[2] / sqrt_mass
            };
            T scaled_momentum[3] = {
                sqrt_mass * g_vel[0],
                sqrt_mass * g_vel[1],
                sqrt_mass * g_vel[2]
            };
            // Instead of taking the max between the impulse scale and the momentum scale, we sum them.
            // This is differs from the result from using max by at most a factor of 2.
            T  momentum_scale = norm_sqr<3>(&scaled_momentum[0]) + norm_sqr<3>(&scaled_impulse[0]);
            atomicAdd(norm_impulse, momentum_scale);

            // M (v - v*) - J^T * γ
            local_Grad[0] += mass * (g_vel[0] - g_v_star[cell_idx * 3 + 0]);
            local_Grad[1] += mass * (g_vel[1] - g_v_star[cell_idx * 3 + 1]);
            local_Grad[2] += mass * (g_vel[2] - g_v_star[cell_idx * 3 + 2]);


            T negative_local_Grad[3] = {
                -local_Grad[0],
                -local_Grad[1],
                -local_Grad[2]
            };

            T scaled_grad[3] = {
                local_Grad[0] / sqrt_mass,
                local_Grad[1] / sqrt_mass,
                local_Grad[2] / sqrt_mass
            };

            // Optimization problem for minimizing ℓ(v_i):
            // min_{v_i} ℓ(v_i) = (1/2) * ||v_i - v_i^*||_M^2 + ℓ_c(v_p(v_i))
            // M * (v_{n+1}^i - v_i^*) = J^T * γ(v_p(v_{n+1}^i))
            // T Hess_Inv[9];
            // inverse3(local_Hess, Hess_Inv);
            // matmul<3, 3, 1, T>(Hess_Inv, negative_local_Grad, local_Dir);
            cholesky_solve3(local_Hess, negative_local_Grad, local_Dir);

            // stop criterion
            atomicAdd(norm_dir, norm_sqr<3>(&scaled_grad[0]));
        }
    }
}

template<typename T, int BLOCK_DIM>
__global__ void grid_to_particle_contact_term_line_search_kernel(
    const GridConfig<T> gconf,
    const size_t n_particles,
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
    T* g_E1,
    T* g_dE1,
    T* g_d2E1,
    const T dt,
    const T friction_mu,
    const T stiffness,
    const T damping,
    const T global_alpha) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // In [Fei et.al 2021],
    // we spill the B-spline weights (nine floats for each thread) by storing them into the shared memory
    // (instead of registers), although none of them is shared between threads. 
    // This choice increases performance, particularly when the number of threads is large (§6.2.5).
    __shared__ T weights[BLOCK_DIM][3][3];

    if (idx < n_particles) {
        uint32_t base[3] = {
            static_cast<uint32_t>(contact_pos[idx * 3 + 0] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 1] * gconf.G_DX_INV - T(0.5)),
            static_cast<uint32_t>(contact_pos[idx * 3 + 2] * gconf.G_DX_INV - T(0.5))
        };
        T fx[3] = {
            contact_pos[idx * 3 + 0] * gconf.G_DX_INV - static_cast<T>(base[0]),
            contact_pos[idx * 3 + 1] * gconf.G_DX_INV - static_cast<T>(base[1]),
            contact_pos[idx * 3 + 2] * gconf.G_DX_INV - static_cast<T>(base[2])
        };
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
            weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
            weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
        }

        // v_p of current jacobi iteration{k}
        T v_p_current[3] = {0, 0, 0};
        // v_p of next jacobi iteration{k+1}
        T v_p_next[3] = {0, 0, 0};

        T vp_search_dir_W[3] = {0, 0, 0};

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    const uint32_t target_cell_index = cell_index(gconf, base[0] + i, base[1] + j, base[2] + k);
                    T weight = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] * weights[threadIdx.x][k][2];

                    const T* g_v = &g_velocities[target_cell_index * 3];
                    const T* g_D = &g_Dir[target_cell_index * 3];
                    v_p_current[0] += weight * g_v[0];
                    v_p_current[1] += weight * g_v[1];
                    v_p_current[2] += weight * g_v[2];
                    v_p_next[0] += weight * (g_v[0] + global_alpha * g_D[0]);
                    v_p_next[1] += weight * (g_v[1] + global_alpha * g_D[1]);
                    v_p_next[2] += weight * (g_v[2] + global_alpha * g_D[2]);
                    vp_search_dir_W[0] += weight * g_D[0];
                    vp_search_dir_W[1] += weight * g_D[1];
                    vp_search_dir_W[2] += weight * g_D[2];
                }
            }
        }

        const T mass = volumes[contact_mpm_id[idx]] * config::DENSITY<T>;
        const T* v_p_n = &velocities[contact_mpm_id[idx] * 3];

        T nhat_W[3] = {contact_normal[idx * 3 + 0], contact_normal[idx * 3 + 1], contact_normal[idx * 3 + 2]};
        T phi0 = -contact_dist[idx];

        T vn_rel_W[3] = {
            v_p_n[0] - contact_rigid_v[idx * 3 + 0],
            v_p_n[1] - contact_rigid_v[idx * 3 + 1],
            v_p_n[2] - contact_rigid_v[idx * 3 + 2]
        };
        T v_current_rel_W[3] = {
            v_p_current[0] - contact_rigid_v[idx * 3 + 0],
            v_p_current[1] - contact_rigid_v[idx * 3 + 1],
            v_p_current[2] - contact_rigid_v[idx * 3 + 2]
        };

        T v_next_rel_W[3] = {
            v_p_next[0] - contact_rigid_v[idx * 3 + 0],
            v_p_next[1] - contact_rigid_v[idx * 3 + 1],
            v_p_next[2] - contact_rigid_v[idx * 3 + 2]
        };

        constexpr int kZAxis = 2;
        T R_WC[9], R_CW[9]; // for each contact pair, Ji = R_CWp * wip
        make_from_one_unit_vector(nhat_W, kZAxis, R_WC);
        transpose<3, 3, T>(R_WC, R_CW);

        T vn_C[3], v_current_C[3], v_next_C[3]; // in the contact local coordinate
        matmul<3, 3, 1, T>(R_CW, vn_rel_W, vn_C);
        matmul<3, 3, 1, T>(R_CW, v_current_rel_W, v_current_C);
        matmul<3, 3, 1, T>(R_CW, v_next_rel_W, v_next_C);

        // lc(v_p(v_i))
        auto lc = [&](const T* v, const T* v0) {
            // frictional component (Lagged Model)
            // lt(v_t) = μ * γn0 * ||v_t||_s
            const T yn0 = max(stiffness * dt * phi0 * (T(1.) - damping * v0[kZAxis]), T(0.));
            const T lt = friction_mu * yn0 * (sqrt(v[0] * v[0] + v[1] * v[1] + config::epsv<T> * config::epsv<T>) - config::epsv<T>); // Eq. 33

            // normal component (Compliant Contact)

            // vˆ = min(−ϕ0 / δt, 1 / d),
            T v_hat = min(phi0 / dt, T(1.) / damping);

            // N(vn) = N+(min(vn, vˆ); f0)
            const T min_vn_v_hat = min(v_hat, v[kZAxis]);

            // N+(vn; ϕ0) = δt k [−vn (ϕ0 + 1/2 δt vn) + d vn²/2 (ϕ0 + 2/3 δt vn)]
            //            = δt k [−ϕ0 vn  - 1/2 δt vn² + 1/2 d ϕ0 vn² + 1/3 d δt vn³]
            //            = δt k [1/3 d δt vn³ + 1/2 (d ϕ0 - δt) vn² - ϕ0 vn]
            //            = ln_A vn³ + ln_B vn² + ln_C vn
            // where N_A = 1/3 δt² k d,
            //       N_B = 1/2 δt k (d ϕ0 - δt) 
            //       N_C = δt k * (-ϕ0)
            const T N_A = T(1. / 3.) * dt * dt * stiffness * damping;
            const T N_B = T(1. / 2.) * dt * stiffness * (-damping * phi0 - dt);
            const T N_C = dt * stiffness * phi0;
            const T N_vn = N_A * min_vn_v_hat * min_vn_v_hat * min_vn_v_hat 
                       + N_B * min_vn_v_hat * min_vn_v_hat 
                       + N_C * min_vn_v_hat;
            
            // ln(vn) = −N(vn),
            const T ln = -N_vn; 

            return lt + ln;
        };

        atomicAdd(g_E1, lc(v_next_C, vn_C));
        T lc_Hess_C[9], lc_Grad_C[3]; // hess and grad in the contact local coordinate
        compute_contact_grad_and_hess(phi0, dt, stiffness, damping, friction_mu, vn_C, v_next_C, lc_Hess_C, lc_Grad_C);
        T global_dir_C[3];
        matmul<3, 3, 1, T>(R_CW, vp_search_dir_W, global_dir_C);
        atomicAdd(g_dE1, dot<3>(lc_Grad_C, global_dir_C));
        T tmp[3];
        matmul<1, 3, 3, T>(global_dir_C, lc_Hess_C, tmp);
        atomicAdd(g_d2E1, dot<3>(tmp, global_dir_C));
    }
}


template<typename T>
__global__ void update_global_inertia_energy_grid_kernel(
    const GridConfig<T> gconf,
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    const T* g_v_star,
    T* g_momentum,
    const T* g_Dir,
    T* global_E1,
    T* global_dE1,
    T* global_d2E1,
    const T global_alpha) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(gconf, cell_idx);
        if (g_masses[cell_idx] > 0) {
            T* g_vel = &g_momentum[cell_idx * 3];
            const T* v_star = &g_v_star[cell_idx * 3];
            const T mass = g_masses[cell_idx];
            const T* Dir = &g_Dir[cell_idx * 3];
            T v_current_rel[3] = {
                g_vel[0] - v_star[0],
                g_vel[1] - v_star[1],
                g_vel[2] - v_star[2]
            };
            T v_next_rel[3] = {
                g_vel[0] + global_alpha * Dir[0] - v_star[0],
                g_vel[1] + global_alpha * Dir[1] - v_star[1],
                g_vel[2] + global_alpha * Dir[2] - v_star[2]
            };

            // (1/2) * ||v_i - v_i^*||_M^2
            atomicAdd(global_E1, T(0.5) * mass * norm_sqr<3>(v_next_rel));
            atomicAdd(global_dE1,  mass * dot<3>(v_next_rel, Dir));
            atomicAdd(global_d2E1, mass * norm_sqr<3>(Dir));
        }
    }
}

template<typename T>
__global__ void apply_global_line_search_grid_kernel(
    const GridConfig<T> gconf,
    const uint32_t touched_cells_cnt,
    uint32_t* g_touched_ids,
    const T* g_masses,
    T* g_momentum,
    const T* g_Dir,
    const T global_alpha) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < touched_cells_cnt) {
        uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
        uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
        uint3 xyz = inverse_cell_index(gconf, cell_idx);
        if (g_masses[cell_idx] > 0) {
            T* g_vel = &g_momentum[cell_idx * 3];
            const T* Dir = &g_Dir[cell_idx * 3];
            g_vel[0] += global_alpha * Dir[0];
            g_vel[1] += global_alpha * Dir[1];
            g_vel[2] += global_alpha * Dir[2];
        }
    }
}

template<typename T>
__global__ void apply_contact_impulse_to_rigid_bodies(
    const size_t n_contacts,
    const T* contact_pos,
    const T* contact_vel_star,
    const T* contact_vel,
    const T* volumes,
    const T* velocities,
    const T* contact_dist,
    const T* contact_normal,
    const T* contact_rigid_v,
    const uint32_t* contact_mpm_id,
    const uint32_t* contact_rigid_id,
    const T* contact_rigid_p_WB,
    T* F_Bq_W_tau,
    T* F_Bq_W_f,
    const T dt,
    const T friction_mu,
    const T stiffness,
    const T damping) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_contacts) {
        // Use negative contact energy gradient as the impulse 
        // instead of particle mdv when accumulating impulses on rigid bodies
        const T mass = volumes[contact_mpm_id[idx]] * config::DENSITY<T>;
        const T* particle_vn = &velocities[contact_mpm_id[idx] * 3];
        const T* particle_v = &contact_vel[idx * 3];

        T nhat_W[3] = {contact_normal[idx * 3 + 0], contact_normal[idx * 3 + 1], contact_normal[idx * 3 + 2]};
        T phi0 = -contact_dist[idx];

        T vn_rel_W[3] = {
            particle_vn[0] - contact_rigid_v[idx * 3 + 0],
            particle_vn[1] - contact_rigid_v[idx * 3 + 1],
            particle_vn[2] - contact_rigid_v[idx * 3 + 2]
        };
        T v_rel_W[3] = {
            particle_v[0] - contact_rigid_v[idx * 3 + 0],
            particle_v[1] - contact_rigid_v[idx * 3 + 1],
            particle_v[2] - contact_rigid_v[idx * 3 + 2]
        };

        constexpr int kZAxis = 2;
        T R_WC[9], R_CW[9]; // for each contact pair, Ji = R_CWp * wip
        make_from_one_unit_vector(nhat_W, kZAxis, R_WC);
        transpose<3, 3, T>(R_WC, R_CW);

        T vn_C[3], v_next_C[3]; // in the contact local coordinate
        matmul<3, 3, 1, T>(R_CW, vn_rel_W, vn_C);
        matmul<3, 3, 1, T>(R_CW, v_rel_W, v_next_C);

        T lc_Hess_C_unused[9], lc_Grad_C[3]; // hess and grad in the contact local coordinate
        compute_contact_grad_and_hess(phi0, dt, stiffness, damping, friction_mu, vn_C, v_next_C, lc_Hess_C_unused, lc_Grad_C);
        
        
        // grad in the world local coordinate
        T lc_Grad_W[3];

        // Grad for lc(vp(vi))
        matmul<3, 3, 1, T>(R_WC, lc_Grad_C, lc_Grad_W); // J^T Grad

        // NOTE (changyu): we have two negative signs here 
        // (1): we negate the sign of the gradient of lc() to get the impulse(gamma) applied to the grid node.
        // (2): we negate (1) again to compute the reaction force to the rigid body, in accordance with Newton's Third Law.
        // finally we have no sign here
        T l_WN_W[3] = {
            lc_Grad_W[0],
            lc_Grad_W[1],
            lc_Grad_W[2]
        };

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