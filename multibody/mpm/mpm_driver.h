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

 private:
  void UpdateParticleStress();

#ifdef NOT_YET_IMPLEMENTED
  void UpdateParticleStress() {
    for (int m = 0; m < ssize(particles_.materials); ++m) {
      const auto& constitutive_model = particles_.constitutive_models[m];
      [[maybe_unused]] const int num_threads = parallelism_.num_threads();
      const int lanes = SimdScalar<T>::lanes();
      std::vector<int> indices.reserve(lanes);
      StrainData<SimdScalar<T>> strain_data;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for (int i = particles_.materials[m].first;
           i < particles_.materials[m].second; i += lanes) {
        const int end = std::min(i + lanes, particles_.materials[m].second);
        indices.resize(end - i);
        std::iota(indices.begin(), indices.end(), i);
        const Matrix3<SimdScalar<T>> F = Load(particles_.F, indices);
        const auto& constitutive_model = particles_.constitutive_models[m];
        strain_data Matrix3<SimdScalar<T>> P;
        constitutive_model.CalcFirstPiolaStress(F);
        particles_.stress[i] = stress;
      }
    }
  }
#endif

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
