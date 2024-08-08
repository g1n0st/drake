#pragma once

#include <array>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/deformation_gradient_data.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Data supporting calculations in StvkHenckyVonMisesModel.
 @tparam_nonsymbolic_scalar. */
template <typename T>
class StvkHenckyVonMisesModelData
    : public DeformationGradientData<StvkHenckyVonMisesModelData<T>> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(StvkHenckyVonMisesModelData);

  /* Constructs a StvkHenckyVonMisesModelData with no deformation. */
  StvkHenckyVonMisesModelData();

  const Matrix3<T>& U() const { return U_; }
  const Matrix3<T>& V() const { return V_; }
  const Vector3<T>& sigma() const { return sigma_; }
  const Vector3<T>& one_over_sigma() const { return one_over_sigma_; }
  const Vector3<T>& log_sigma() const { return log_sigma_; }
  const T& log_sigma_trace() const { return log_sigma_trace_; }

  Vector3<T>& mutable_sigma() { return sigma_; }
  Vector3<T>& mutable_one_over_sigma() { return one_over_sigma_; }
  Vector3<T>& mutable_log_sigma() { return log_sigma_; }
  T& mutable_log_sigma_trace() { return log_sigma_trace_; }

 private:
  friend DeformationGradientData<StvkHenckyVonMisesModelData<T>>;

  /* Shadows DeformationGradientData::UpdateFromDeformationGradient() as
   required by the CRTP base class. */
  void UpdateFromDeformationGradient();

  /* F = UΣVᵀ is the singular value decomposition of the deformation gradient.
   */
  Matrix3<T> U_;
  Matrix3<T> V_;
  Vector3<T> sigma_;
  Vector3<T> one_over_sigma_;
  Vector3<T> log_sigma_;
  T log_sigma_trace_{};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
