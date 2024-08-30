#pragma once

#include "particles.h"
#include "sparse_grid.h"

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/parallelism.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/multibody/fem/deformable_body_config.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
class MpmDriver {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmDriver);

  MpmDriver(T dt, T dx, int num_subteps, Parallelism parallelism = false)
      : dt_(dt),
        num_subteps_(num_subteps),
        substep_dt_(dt / num_subteps),
        dx_(dx),
        grid_(std::make_unique<SparseGrid<T>>(dx, parallelism)),
        parallelism_(parallelism) {
    DRAKE_THROW_UNLESS(num_subteps > 0);
    DRAKE_THROW_UNLESS(dt > 0);
    DRAKE_THROW_UNLESS(dx > 0);
  }

  /* Sample particles inside the given geometry.
   @param[in] geometry_instance The geometry instance to sample particles
   inside. Only the shape and pose of the geometry is used; all the geometry
   properties are discarded.
   @param[in] particles_per_cell The targeted number of particle to be sampled
   in each grid cell (of size dx * dx * dx).
   @param[in] config  The physical properties of the material. */
  void SampleParticles(
      std::unique_ptr<geometry::GeometryInstance> geometry_instance,
      int particles_per_cell, const fem::DeformableBodyConfig<T>& config);

  void AdvanceOneTimeStep(
      const geometry::QueryObject<double>& query_object,
      const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
      const std::vector<math::RigidTransform<double>>& poses,
      const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
          geometry_id_to_body_index);

  void UpdateContactForces(
      const geometry::QueryObject<double>& query_object,
      const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
      const std::vector<math::RigidTransform<double>>& poses,
      const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
          geometry_id_to_body_index);

  const ParticleData<T>& particles() const { return particles_; }

  const std::vector<multibody::ExternallyAppliedSpatialForce<double>>&
  rigid_forces() const {
    return rigid_forces_;
  }

 private:
  // TODO(xuchenhan-tri): Move these to the particles class.
  void UpdateParticleStress();
  void SimdUpdateParticleStress();

  T dt_{0.0};
  int num_subteps_{0};
  T substep_dt_{0.0};
  T dx_{0.0};
  Vector3<T> gravity_{0, 0, -9.81};
  copyable_unique_ptr<SparseGrid<T>> grid_;
  ParticleData<T> particles_;
  Parallelism parallelism_;
  std::vector<multibody::ExternallyAppliedSpatialForce<double>> rigid_forces_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
