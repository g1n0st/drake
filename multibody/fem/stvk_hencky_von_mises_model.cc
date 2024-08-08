#include "drake/multibody/fem/stvk_hencky_von_mises_model.h"

#include <array>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/multibody/fem/calc_lame_parameters.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
StvkHenckyVonMisesModel<T>::StvkHenckyVonMisesModel(const T& youngs_modulus,
                                                    const T& poissons_ratio,
                                                    const T& yield_stress)
    : E_(youngs_modulus), nu_(poissons_ratio), yield_stress_(yield_stress) {
  const LameParameters<T> lame_params = CalcLameParameters(E_, nu_);
  mu_ = lame_params.mu;
  lambda_ = lame_params.lambda;
}

template <typename T>
void StvkHenckyVonMisesModel<T>::CalcElasticEnergyDensityImpl(const Data& data,
                                                              T* Psi) const {
  /* Psi = μ tr((log Σ)^2) + 1/2 λ (tr(log Σ))^2 */
  *Psi = mu_ * data.log_sigma().squaredNorm() +
         0.5 * lambda_ * data.log_sigma_trace() * data.log_sigma_trace();
}

template <typename T>
void StvkHenckyVonMisesModel<T>::CalcFirstPiolaStressImpl(const Data& data,
                                                          Matrix3<T>* P) const {
  /* P = U (2 μ Σ^{-1} (log Σ) + λ tr(log Σ) Σ^{-1}) V^T */
  Vector3<T> P_hat = 2 * mu_ * data.log_sigma() +
                     lambda_ * data.log_sigma_trace() * Vector3<T>::Ones();
  P_hat = data.one_over_sigma().asDiagonal() * P_hat;

  (*P) = data.U() * P_hat.asDiagonal() * data.V().transpose();
}

template <typename T>
void StvkHenckyVonMisesModel<T>::CalcFirstPiolaStressDerivativeImpl(
    const Data&, Eigen::Matrix<T, 9, 9>*) const {
  throw std::runtime_error("Not implemented.");
}

template <typename T>
void StvkHenckyVonMisesModel<T>::ProjectStrainImpl(Matrix3<T>* F,
                                                   Data* data) const {
  if (!(yield_stress_ > 0.0)) return;
  data->UpdateData(*F, *F);
  /* The deviatoric component of Kirchhoff stress in the principal frame is:
   dev(τ) = τ - pJI = τ - 1/3tr(τ)I
   In the principal frame: dev(τ) = 2μlog(σᵢ) - 2μ/3 ∑ᵢ log(σᵢ) =
   2μ[log(σᵢ)-1/3*∑ᵢ log(σᵢ)] */
  Vector3<T> deviatoric_tau =
      2.0 * mu_ *
      (data->log_sigma() -
       1.0 / 3.0 * data->log_sigma_trace() * Vector3<T>::Ones());
  const T yield_function = kSqrt3Over2 * deviatoric_tau.norm() - yield_stress_;
  if (yield_function > 0) {
    /* Note that deviatoric_tau is nonzero because both yield_stress_ and
     yield_function are positive. */
    const Vector3<T> nu = kSqrt3Over2 * deviatoric_tau.normalized();
    const T delta_gamma = yield_function / (3.0 * mu_);
    /* Update the singular values of Hencky strain ε. */
    data->mutable_log_sigma() = data->log_sigma() - delta_gamma * nu;
    /* exp(ε) gives the deformation gradient in the principal frame. */
    data->mutable_sigma() = (data->log_sigma()).array().exp();
    const Vector3<T>& s = data->sigma();
    const T kEps = 16.0 * std::numeric_limits<T>::epsilon();
    data->mutable_one_over_sigma() =
        Vector3<T>(1.0 / std::max(s(0), kEps), 1.0 / std::max(s(1), kEps),
                   1.0 / std::max(s(2), kEps));
    data->mutable_log_sigma_trace() = data->log_sigma().array().sum();
    /* Get the full F. */
    *F = data->U() * data->sigma().asDiagonal() * data->V().transpose();
  }
}

template class StvkHenckyVonMisesModel<double>;
template class StvkHenckyVonMisesModel<float>;
template class StvkHenckyVonMisesModel<AutoDiffXd>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
