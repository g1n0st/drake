#include "mpm_driver.h"

#include <iostream>
#include <variant>

#include "transfer.h"

#include "drake/common/ssize.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/mpm/poisson_disk.h"
#include "drake/multibody/plant/contact_properties.h"
#include "drake/multibody/plant/coulomb_friction.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {
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
}  // namespace

using drake::geometry::SignedDistanceToPoint;
using Eigen::Vector3d;
using fem::DeformableBodyConfig;
using geometry::Box;
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

  void ImplementGeometry(const Box& box, void* data) override {
    DRAKE_ASSERT(data != nullptr);
    BoundingBox& result = *static_cast<BoundingBox*>(data);
    for (int d = 0; d < 3; ++d) {
      result[0][d] = -box.size()(d) / 2.0;
      result[1][d] = box.size()(d) / 2.0;
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
    case fem::MaterialModel::kStvkHenckyVonMises:
      return fem::internal::StvkHenckyVonMisesModel<T>(config.youngs_modulus(),
                                                       config.poissons_ratio(),
                                                       config.yield_stress());
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
  std::vector<Vector3<double>> q_GPs =
      FilterPoints(q_GP_candidates, geometry_instance->shape());
  grid_->SortParticles(&q_GPs);
  const int num_particles = ssize(q_GPs);
  const T mass_density = config.mass_density();
  const double total_volume = geometry::CalcVolume(geometry_instance->shape());
  const T volume_per_particle = total_volume / num_particles;
  const RigidTransform<double>& X_WG = geometry_instance->pose();
  const int num_existing_particles = ssize(particles_.m);
  ConstitutiveModelVariant<T> constitutive_model =
      MakeConstitutiveModel<T>(config);
  for (int i = 0; i < num_particles; ++i) {
    const Vector3<double> q_WP = X_WG * q_GPs[i];
    particles_.m.push_back(mass_density * volume_per_particle);
    particles_.x.push_back(q_WP.cast<T>());
    particles_.v.push_back({0, 0, 0});
    particles_.F.push_back(Matrix3<T>::Identity());
    particles_.tau_v0.push_back(Matrix3<T>::Zero());
    particles_.C.push_back(Matrix3<T>::Zero());
    particles_.volume.push_back(volume_per_particle);
    std::visit(
        [this](auto& model) {
          particles_.strain_data.push_back(model.MakeDefaultData());
        },
        constitutive_model);
  }
  particles_.constitutive_models.push_back(constitutive_model);
  particles_.materials.push_back(
      {num_existing_particles, num_existing_particles + num_particles});
}

template <typename T>
void MpmDriver<T>::AdvanceOneTimeStep(
    const geometry::QueryObject<double>& query_object,
    const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
    const std::vector<math::RigidTransform<double>>& poses,
    const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
        geometry_id_to_body_index) {
  rigid_forces_.resize(poses.size());
  for (int i = 0; i < ssize(rigid_forces_); ++i) {
    /* We use `p_BoBq_B` to temporarily store p_WB. We will replace it with the
     actual value of p_BoBq_B later on. */
    auto& force = rigid_forces_[i];
    force.body_index = BodyIndex(i);
    force.p_BoBq_B = poses[i].translation();
    force.F_Bq_W.SetZero();
  }
  for (int i = 0; i < num_subteps_; ++i) {
    UpdateParticleStress();
    /* Update particle's contact (and friction) momentum and accumulate the
     opposite momentum in rigid_forces_. */
    UpdateContactForces(query_object, spatial_velocities, poses,
                        geometry_id_to_body_index);
    // Particle to grid transfer.
    Transfer<T> transfer(substep_dt_, grid_.get_mutable(), &particles_);
    transfer.ParallelSimdParticleToGrid(parallelism_);
    // Grid velocity update.
    grid_->ExplicitVelocityUpdate(Vector3<T>::Zero());
    // Grid to particle transfer.
    transfer.ParallelSimdGridToParticle(parallelism_);
  }
  /* Restore p_BoBq_B value and divide by dt to turn impulse into forces. */
  for (int i = 0; i < ssize(rigid_forces_); ++i) {
    rigid_forces_[i].p_BoBq_B = Vector3<double>::Zero();
    rigid_forces_[i].F_Bq_W.rotational() /= dt_;
    rigid_forces_[i].F_Bq_W.translational() /= dt_;
  }
}

template <typename T>
void MpmDriver<T>::UpdateContactForces(
    const geometry::QueryObject<double>& query_object,
    const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
    const std::vector<math::RigidTransform<double>>& poses,
    const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
        geometry_id_to_body_index) {
  const double kStiffness = 1e6;
  const double kDamping = 1.0;
  const double substep_dt = dt_ / double(num_subteps_);
  for (auto& v : particles_.v) {
    v += gravity_.template cast<T>() * substep_dt;
  }
  ContactForceSolver<double> solver(substep_dt, kStiffness, kDamping);

  // TODO(xuchenhan-tri): Run this in parallel. Be careful about the race
  // condition.
  for (int p = 0; p < ssize(particles_.m); ++p) {
    const Vector3<double>& p_WP = particles_.x[p].template cast<double>();
    const std::vector<SignedDistanceToPoint<double>>& signed_distances =
        query_object.ComputeSignedDistanceToPoint(p_WP, 0.001);
    for (const SignedDistanceToPoint<double>& sd : signed_distances) {
      const double& phi = -sd.distance;
      if (phi < 0) continue;
      const int body_index = geometry_id_to_body_index.at(sd.id_G);
      const CoulombFriction<double>& coulomb_friction =
          multibody::internal::GetCoulombFriction(sd.id_G,
                                                  query_object.inspector());
      double mu = coulomb_friction.dynamic_friction();
      const Vector3<double> nhat_W = sd.grad_W.normalized();
      /* World frame position of the origin of the rigid body. */
      const Vector3<double>& p_WR = poses[body_index].translation();
      const Vector3<double> p_RP = p_WP - p_WR;
      /* World frame velocity of a point affixed to the rigid body that
       coincide with the particle. */
      const Vector3<double> v_WRp =
          spatial_velocities[body_index].Shift(p_RP).translational();
      const Vector3<double> vc =
          particles_.v[p].template cast<double>() - v_WRp;  // relative velocity
      const double vn = vc.dot(nhat_W);
      const double mp = particles_.m[p];
      const double vn_next = solver.Solve(mp, vn, phi);
      if (vn_next != vn) {
        const Vector3<double> vt = vc - vn * nhat_W;
        double dvn = vn_next - vn;
        /* The velocity change at the particle. */
        Vector3<double> dv = dvn * nhat_W;
        const double vt_norm = vt.norm();
        /* Safely normalize the tangent vector. */
        Vector3<double> vt_hat = Vector3<double>::Zero();
        if (vt_norm > 1e-10) {
          vt_hat = vt / vt_norm;
        }
        /* kf is the slope of the regulated friction in stiction. Larger kf
         resolves static friction better, but is less numerically stable. */
        const double kf = 10.0;
        dv -= std::min(dvn * mu, kf * vt_norm) * vt_hat;

        particles_.v[p] += dv.template cast<T>();
        /* We negate the sign of the particles momentum change to get
         the impulse applied to the rigid body at the grid node. */
        const Vector3d l_WR_W = -mp * dv;
        const Vector3d& p_WR = rigid_forces_.at(body_index).p_BoBq_B;
        const Vector3d p_RP_W = p_WP - p_WR;
        /* The angular impulse applied to the rigid body at the grid
         node. */
        const Vector3d h_WPRo_W = p_RP_W.cross(l_WR_W);
        /* Use `F_Bq_W` to store the spatial impulse applied to the body
         at its origin, expressed in the world frame. */
        rigid_forces_.at(body_index).F_Bq_W +=
            SpatialForce<double>(h_WPRo_W, l_WR_W);
      }
    }
  }
}

template <typename T>
void MpmDriver<T>::UpdateParticleStress() {
  for (int m = 0; m < ssize(particles_.materials); ++m) {
    const auto& constitutive_model = particles_.constitutive_models[m];
    [[maybe_unused]] const int num_threads = parallelism_.num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = particles_.materials[m].first;
         i < particles_.materials[m].second; ++i) {
      std::visit(
          [&, this](auto& model) {
            Matrix3<T>& F = particles_.F[i];
            using StrainDataType = typename std::decay_t<decltype(model)>::Data;
            StrainDataType& strain_data =
                std::get<StrainDataType>(particles_.strain_data[i]);
            model.ProjectStrain(&F, &strain_data);
            model.CalcFirstPiolaStress(strain_data, &particles_.tau_v0[i]);
            particles_.tau_v0[i] *= particles_.volume[i] * F.transpose();
          },
          constitutive_model);
    }
  }
}

template <typename T>
void MpmDriver<T>::SimdUpdateParticleStress() {
  for (int m = 0; m < ssize(particles_.materials); ++m) {
    const auto& constitutive_model = particles_.constitutive_models[m];
    [[maybe_unused]] const int num_threads = parallelism_.num_threads();
    const int lanes = SimdScalar<T>::lanes();
    std::vector<int> indices(lanes);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = particles_.materials[m].first;
         i < particles_.materials[m].second; i += lanes) {
      const int end = std::min(i + lanes, particles_.materials[m].second);
      indices.resize(end - i);
      std::iota(indices.begin(), indices.end(), i);
      const Matrix3<SimdScalar<T>> F = Load(particles_.F, indices);
      const SimdScalar<T> volume = Load(particles_.volume, indices);
      std::visit(
          [&, this](auto& model) {
            const Matrix3<SimdScalar<T>> P = model.CalcFirstPiolaStress(F);
            const Matrix3<SimdScalar<T>> tau_v0 = volume * P * F.transpose();
            particles_.tau_v0[0].setZero();
            Store(tau_v0, &particles_.tau_v0, indices);
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