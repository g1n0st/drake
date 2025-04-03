#pragma once

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/settings.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

#include "multibody/gpu_mpm/tph_poisson.h"
#pragma GCC diagnostic pop

namespace drake {
namespace multibody {
namespace gmpm {

// NOTE(changyu): `MpmConfigParams` is responsive to store the initial config parameters in `CpuMpmModel`,
template<typename T = config::GpuT>
struct MpmConfigParams {
    // grid domain parameters
    int domain_bits {7};
    T grid_block_spacing {static_cast<T>(1.)};

    // material parameters
    T density {static_cast<T>(1500.0)};

    T youngs_modules {static_cast<T>(4e5)};
    T poisson_ratio {static_cast<T>(.3)};

    // particle-mpm properties
    T particle_yield_stress {static_cast<T>(1e9)};
    T particle_plasticity {true};
    T particle_linear_corotated {false};

    // cloth-mpm properties
    T cloth_gamma {static_cast<T>(0.)};
    T cloth_K {static_cast<T>(100000.)};
    T cloth_c_F {static_cast<T>(0.)};
    T cloth_thickness {static_cast<T>(0.001)};

    T sdf_friction {static_cast<T>(.5)};
    T rpic_damping {static_cast<T>(.0)};

    T substep_dt {static_cast<T>(1e-3)};
    bool write_files {false};
    T contact_stiffness {static_cast<T>(1e5)};
    T contact_epsv {static_cast<T>(1e-3)};
    T contact_damping {static_cast<T>(0.0)};
    T contact_friction_mu {static_cast<T>(0.0)};
    int mpm_bc {-1};
    bool ignore_face_contact {false};
    T margin {static_cast<T>(1.0)};
};

template<typename T = config::GpuT>
inline std::vector<Vec3<T>> sample_particle_mpm_box(const T minx[3], const T maxx[3], const T ppc, const T dx) {
    const tph_poisson_real bounds_min[3] = { 
        static_cast<tph_poisson_real>(minx[0]), static_cast<tph_poisson_real>(minx[1]), static_cast<tph_poisson_real>(minx[2])
    };
    const tph_poisson_real bounds_max[3] = { 
        static_cast<tph_poisson_real>(maxx[0]), static_cast<tph_poisson_real>(maxx[1]), static_cast<tph_poisson_real>(maxx[2])
    };

    const tph_poisson_real h = dx;
    const tph_poisson_real sample_r = h / (std::cbrt(tph_poisson_real(ppc)) + 1);
    
    const tph_poisson_args args = { 
        .bounds_min = bounds_min,
        .bounds_max = bounds_max,
        .seed = UINT64_C(666),
        .radius = sample_r,
        .ndims = INT32_C(3),
        .max_sample_attempts = UINT32_C(30)
    };

    const tph_poisson_allocator *alloc = NULL;
    tph_poisson_sampling sampling;
    memset(&sampling, 0, sizeof(tph_poisson_sampling));

    const int ret = tph_poisson_create(&args, alloc, &sampling);
    if (ret != TPH_POISSON_SUCCESS) {
        throw;
    }

    const tph_poisson_real *samples = tph_poisson_get_samples(&sampling);
    if (samples == NULL) {
        throw;
    }

    std::vector<Vec3<T>> pts;
    for (int i = 0; i < sampling.nsamples; ++i) {
        pts.push_back(Vec3<T>(
            T(samples[i * 3 + 0]),
            T(samples[i * 3 + 1]),
            T(samples[i * 3 + 2])
        ));
    }

    return pts;
}

// NOTE(changyu): `CpuMpmModel` is responsive to store the initial config in `DeformableModel`,
// (mesh topology, particle state, material/solver parameters, etc.),
// and use them to initialize the `GpuMpmState` when all finalize.
// TODO (changyu): now it only be specific for MPM cloth.
template<typename T>
struct CpuMpmModel {
    CpuMpmModel() = default;
    std::vector<Vec3<T>> pos;
    std::vector<Vec3<T>> vel;
    std::vector<T> vol; // only for particle-mpm
    std::vector<int> indices; // only for cloth-mpm

    MpmConfigParams<T> config;
};

// NOTE(changyu): a temporary data buffer used to do the communication between
// `DeformableModel` input port and `DrakeVisualizer` output port to visualize MPM-related data.
// TODO (changyu): now it only be specific for MPM cloth.
template<typename T>
struct MpmPortData {
    std::vector<Vec3<T>> pos;
    std::vector<int> indices;
};

// NOTE (changyu): from Zeshun's code.
/* Stores all info about mpm particles that are in contact with rigid bodies (defined to be 
particles that fall within rigid bodies).

                  Mpm Particles (endowed with an ordering)

            
            `1    `2    `3    `4
            
                             ---------
            `5    `6    `7   |*8
                             |      Rigid body with id B
      ----------             |
            *9 |  `10   `11  |*12
               |             ---------
               |
Rigid body with id A

*: particles in contact
`: particles not in contact
 */

template <typename T>
struct MpmParticleContactPairs {
   std::vector<uint32_t> particle_in_contact_index;
   std::vector<uint32_t> non_mpm_id;
   std::vector<T> penetration_distance;
   std::vector<Vec3<T>> normal;
   std::vector<Vec3<T>> particle_in_contact_position;
   std::vector<Vec3<T>> rigid_v;
   std::vector<Vec3<T>> rigid_p_WB;

   void clear() {
        particle_in_contact_index.clear();
        non_mpm_id.clear();
        penetration_distance.clear();
        normal.clear();
        particle_in_contact_position.clear();
        rigid_v.clear();
        rigid_p_WB.clear();
   }

   void push_back(
    uint32_t particle_in_contact_index_,
    uint32_t non_mpm_id_,
    T penetration_distance_,
    Vec3<T> normal_,
    Vec3<T> particle_in_contact_position_,
    Vec3<T> rigid_v_,
    Vec3<T> rigid_p_WB_
    ) {
        particle_in_contact_index.push_back(particle_in_contact_index_);
        non_mpm_id.push_back(non_mpm_id_);
        penetration_distance.push_back(penetration_distance_);
        normal.push_back(normal_);
        particle_in_contact_position.push_back(particle_in_contact_position_);
        rigid_v.push_back(rigid_v_);
        rigid_p_WB.push_back(rigid_p_WB_);
   }

   size_t size() const {
      return non_mpm_id.size();
   }
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake