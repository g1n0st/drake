#pragma once

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/cuda_mpm_model.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

// NOTE(changyu): `MpmConfigParams` is responsive to store the initial config parameters in `CpuMpmModel`,
template<typename T = config::GpuT>
struct MpmConfigParams {
    T substep_dt {static_cast<T>(1e-3)};
    bool write_files {false};
    T contact_stiffness{static_cast<T>(1e5)};
    T contact_damping{static_cast<T>(0.0)};
    T contact_friction_mu{static_cast<T>(0.0)};
    int contact_query_frequency{1};
    int mpm_bc{-1};
};

// NOTE(changyu): `CpuMpmModel` is responsive to store the initial config in `DeformableModel`,
// (mesh topology, particle state, material/solver parameters, etc.),
// and use them to initialize the `GpuMpmState` when all finalize.
// TODO (changyu): now it only be specific for MPM cloth.
template<typename T>
struct CpuMpmModel {
    CpuMpmModel() = default;
    std::vector<Vec3<T>> cloth_pos;
    std::vector<Vec3<T>> cloth_vel;
    std::vector<int> cloth_indices;

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
   std::vector<int64_t> non_mpm_id;
   std::vector<T> penetration_distance;
   std::vector<Vec3<T>> normal;
   std::vector<Vec3<T>> particle_in_contact_position;
   std::vector<Vec3<T>> rigid_v;

   void clear() {
        particle_in_contact_index.clear();
        non_mpm_id.clear();
        penetration_distance.clear();
        normal.clear();
        particle_in_contact_position.clear();
        rigid_v.clear();
   }

   void push_back(
    uint32_t particle_in_contact_index_,
    int64_t non_mpm_id_,
    T penetration_distance_,
    Vec3<T> normal_,
    Vec3<T> particle_in_contact_position_,
    Vec3<T> rigid_v_
    ) {
        particle_in_contact_index.push_back(particle_in_contact_index_);
        non_mpm_id.push_back(non_mpm_id_);
        penetration_distance.push_back(penetration_distance_);
        normal.push_back(normal_);
        particle_in_contact_position.push_back(particle_in_contact_position_);
        rigid_v.push_back(rigid_v_);
   }

   size_t size() const {
      return non_mpm_id.size();
   }
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake