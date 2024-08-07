#pragma once

#include "particles.h"
#include "sparse_grid.h"

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
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmDriver);

  MpmDriver(T dt, T dx, Parallelism parallelism = false)
      : dt_(dt), dx_(dx), grid_(dx, parallelism), parallelism_(parallelism) {}

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

  void AdvanceOneTimeStep();

  const ParticleData<T>& particles() const { return particles_; }

 private:
  void UpdateParticleStress();

  void SimdUpdateParticleStress();

  T dt_{0.0};
  T dx_{0.0};
  Vector3<T> gravity_{0, 0, -9.81};
  SparseGrid<T> grid_;
  ParticleData<T> particles_;
  Parallelism parallelism_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
