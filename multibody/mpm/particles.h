#pragma once

#include <variant>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/fem/linear_constitutive_model.h"
#include "drake/multibody/fem/linear_corotated_model.h"
#include "drake/multibody/mpm/math.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Data structure for conveniently accessing/modifying the attributes of a
 single particle. */
template <typename T>
struct Particle {
  Particle(T* m_in, Vector3<T>* x_in, Vector3<T>* v_in, Matrix3<T>* F_in,
           Matrix3<T>* C_in, Matrix3<T>* tau_v0_in)
      : m(*m_in), x(*x_in), v(*v_in), F(*F_in), C(*C_in), tau_v0(*tau_v0_in) {}
  T& m;
  Vector3<T>& x;
  Vector3<T>& v;
  Matrix3<T>& F;
  Matrix3<T>& C;
  Matrix3<T>& tau_v0;
};

template <typename T>
using ConstitutiveModelVariant =
    std::variant<fem::internal::CorotatedModel<T>,
                 fem::internal::LinearCorotatedModel<T>,
                 fem::internal::LinearConstitutiveModel<T>>;

/* The collection of all physical attributes we care about for all particles.
 All quantities are measured and expressed in the world frame (when
 applicable).
 @tparam double or float. */
template <typename T>
struct ParticleData {
  Particle<T> particle(int i) {
    return Particle<T>(&m[i], &x[i], &v[i], &F[i], &C[i], &tau_v0[i]);
  }

  std::vector<T> m;           // mass
  std::vector<Vector3<T>> x;  // positions
  std::vector<Vector3<T>> v;  // velocity
  std::vector<Matrix3<T>> F;  // deformation gradient
  std::vector<Matrix3<T>> C;  // affine velocity field
  std::vector<Matrix3<T>>
      tau_v0;             // Kirchhoff stress scaled by reference volume
  std::vector<T> volume;  // reference volume

  std::vector<ConstitutiveModelVariant<T>> constitutive_models;
  std::vector<std::pair<int, int>>
      materials;  // Suppose materials[k] = (i, j), then particles with indices
                  // in [i, j) have the same material: constitutive_models[k].
};

template <typename T>
MassAndMomentum<T> ComputeTotalMassAndMomentum(const ParticleData<T>& particles,
                                               const T& dx) {
  MassAndMomentum<T> result;
  const T D = dx * dx * 0.25;
  const int num_particles = particles.m.size();
  for (int i = 0; i < num_particles; ++i) {
    result.mass += particles.m[i];
    result.linear_momentum += particles.m[i] * particles.v[i];
    const Matrix3<T> B = particles.C[i] * D;  // C = B * D^{-1}
    result.angular_momentum +=
        particles.m[i] *
        (particles.x[i].cross(particles.v[i]) + ContractWithLeviCivita(B));
  }
  return result;
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
