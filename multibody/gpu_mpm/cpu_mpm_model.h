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
template <typename T>
class ContactForceSolver {
 public:
  ContactForceSolver(T dt, T k, T d) : dt_(dt), k_(k), d_(d) {}
  // TODO(xuchenhan-tri): Take in the entire velocity vector and return the
  // next velocity (vector) after treating friction.
  T Solve(T m, T v0, T phi0) {
    T v_hat = std::min(phi0 / dt_, 1 / d_);
    if (v0 > v_hat) return v0;
    T a = k_ * d_ * dt_ * dt_;
    T b = -m - (k_ * dt_ * (dt_ + d_ * phi0));
    T c = k_ * dt_ * phi0 + m * v0;
    T discriminant = b * b - 4.0 * a * c;
    T v_next = (-b - std::sqrt(discriminant)) / (2.0 * a);
    return v_next;
  }

 private:
  T dt_;
  T k_;
  T d_;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake