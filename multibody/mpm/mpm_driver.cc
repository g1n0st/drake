#include "mpm_driver.h"

#include <variant>

#include "transfer.h"

#include "drake/common/ssize.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/mpm/poisson_disk.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using fem::DeformableBodyConfig;
using geometry::GeometryInstance;
using geometry::Shape;
using geometry::ShapeReifier;
using geometry::Sphere;
using math::RigidTransform;

class BoundingBoxCalculator : public ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BoundingBoxCalculator);

  BoundingBoxCalculator() = default;

  using BoundingBox = std::array<std::array<double, 3>, 2>;

  BoundingBox Compute(const Shape& shape) {
    BoundingBox result;
    shape.Reify(this, &result);
    return result;
  }

  void ImplementGeometry(const Sphere& sphere, void* data) override {
    DRAKE_ASSERT(data != nullptr);
    BoundingBox& result = *static_cast<BoundingBox*>(data);
    for (int d = 0; d < 3; ++d) {
      result[0][d] = -sphere.radius();
      result[1][d] = sphere.radius();
    }
  }
};

template <typename T>
ConstitutiveModelVariant<T> MakeConstitutiveModel(
    const DeformableBodyConfig<T>& config) {
  switch (config.material_model()) {
    case fem::MaterialModel::kCorotated:
      return fem::internal::CorotatedModel<T>(config.youngs_modulus(),
                                              config.poissons_ratio());
    case fem::MaterialModel::kLinearCorotated:
      return fem::internal::LinearCorotatedModel<T>(config.youngs_modulus(),
                                                    config.poissons_ratio());
    case fem::MaterialModel::kLinear:
      return fem::internal::LinearConstitutiveModel<T>(config.youngs_modulus(),
                                                       config.poissons_ratio());
  }
  DRAKE_UNREACHABLE();
}

template <typename T>
void MpmDriver<T>::SampleParticles(
    std::unique_ptr<GeometryInstance> geometry_instance, int particles_per_cell,
    const DeformableBodyConfig<T>& config) {
  DRAKE_THROW_UNLESS(geometry_instance != nullptr);
  DRAKE_THROW_UNLESS(particles_per_cell > 0);

  BoundingBoxCalculator calculator;
  const std::array<std::array<double, 3>, 2> bounding_box =
      calculator.Compute(geometry_instance->shape());
  const double sampling_radius =
      dx_ / std::cbrt(particles_per_cell * 4.0 / 3.0 * M_PI);
  /* Sample the particles in the geometry's bounding box. */
  const std::vector<Vector3<double>> q_GP_candidates =
      PoissonDiskSampling<double>(sampling_radius, bounding_box[0],
                                  bounding_box[1]);
  /* Reject points that fall outside of the shape. */
  const std::vector<Vector3<double>> q_GPs =
      FilterPoints(q_GP_candidates, geometry_instance->shape());
  const int num_particles = ssize(q_GPs);
  const T mass_density = config.mass_density();
  const double total_volume = geometry::CalcVolume(geometry_instance->shape());
  const T volume_per_particle = total_volume / num_particles;
  const RigidTransform<double>& X_WG = geometry_instance->pose();
  const int num_existing_particles = ssize(particles_.m);
  for (int i = 0; i < num_particles; ++i) {
    // TODO(xuchenhan-tri): Reject the particle if it is outside the geometry.
    const Vector3<double> q_WP = X_WG * q_GPs[i].cast<double>();
    particles_.m.push_back(mass_density * volume_per_particle);
    particles_.x.push_back(q_WP.cast<T>());
    particles_.v.push_back({0, 0, 0});
    particles_.F.push_back(Matrix3<T>::Identity());
    particles_.tau_v0.push_back(Matrix3<T>::Zero());
    particles_.C.push_back(Matrix3<T>::Zero());
    particles_.volume.push_back(volume_per_particle);
  }
  particles_.constitutive_models.push_back(MakeConstitutiveModel<T>(config));
  particles_.materials.push_back(
      {num_existing_particles, num_existing_particles + num_particles});
}

template <typename T>
void MpmDriver<T>::AdvanceOneTimeStep() {
  UpdateParticleStress();
  // Particle to grid transfer.
  Transfer<T> transfer(dt_, &grid_, &particles_);
  transfer.ParallelSimdParticleToGrid(parallelism_);
  // Grid velocity update.
  grid_.ExplicitVelocityUpdate(dt_ * gravity_);
  // TODO(xuchenhan-tri): Add boundary conditions/contact.
  // Grid to particle transfer.
  transfer.ParallelSimdGridToParticle(parallelism_);
}

template <typename T>
void MpmDriver<T>::UpdateParticleStress() {
  for (int m = 0; m < ssize(particles_.materials); ++m) {
    const auto& constitutive_model = particles_.constitutive_models[m];
    for (int i = particles_.materials[m].first;
         i < particles_.materials[m].second; ++i) {
      std::visit(
          [=, this](auto& model) {
            const Matrix3<T>& F = particles_.F[i];
            const Matrix3<T> P = model.CalcFirstPiolaStress(F);
            particles_.tau_v0[i] = particles_.volume[i] * P * F.transpose();
          },
          constitutive_model);
    }
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MpmDriver<double>;
template class drake::multibody::mpm::internal::MpmDriver<float>;