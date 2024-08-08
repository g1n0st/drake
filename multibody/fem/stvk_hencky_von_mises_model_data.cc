#include "drake/multibody/fem/stvk_hencky_von_mises_model_data.h"

#include "drake/common/autodiff.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
StvkHenckyVonMisesModelData<T>::StvkHenckyVonMisesModelData() {
  U_ = Matrix3<T>::Identity();
  V_ = Matrix3<T>::Identity();
  sigma_ = Vector3<T>::Ones();
  one_over_sigma_ = Vector3<T>::Ones();
  log_sigma_ = Vector3<T>::Zero();
  log_sigma_trace_ = 0;
}

template <typename T>
void StvkHenckyVonMisesModelData<T>::UpdateFromDeformationGradient() {
  const Matrix3<T>& F = this->deformation_gradient();
  Eigen::JacobiSVD<Matrix3<T>> svd(F,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  U_ = svd.matrixU();
  V_ = svd.matrixV();
  sigma_ = svd.singularValues();
  const T kEps = 16.0 * std::numeric_limits<T>::epsilon();
  one_over_sigma_ =
      Vector3<T>(1.0 / std::max(sigma_(0), kEps), 1.0 / std::max(sigma_(1), kEps),
                 1.0 / std::max(sigma_(2), kEps));
  log_sigma_ = sigma_.array().log();
  log_sigma_trace_ = log_sigma_.sum();
}

template class StvkHenckyVonMisesModelData<double>;
template class StvkHenckyVonMisesModelData<float>;
template class StvkHenckyVonMisesModelData<AutoDiffXd>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
