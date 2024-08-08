#pragma once

#include <array>

#include "drake/multibody/fem/constitutive_model.h"
#include "drake/multibody/fem/stvk_hencky_von_mises_model_data.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Traits for StvkHenckyVonMisesModel. */
template <typename T>
struct StvkHenckyVonMisesModelTraits {
  using Scalar = T;
  using Data = StvkHenckyVonMisesModelData<T>;
  static constexpr int is_linear = false;
};

/* Implements Saint-Venant Kirchhoff model, but replaces the left Cauchy Green
strain with the Hencky strain https://dl.acm.org/doi/abs/10.1145/2897824.2925906

 @tparam_nonsymbolic_scalar

 [Stomakhin, 2012] Stomakhin, Alexey, et al. "Energetically consistent
 invertible elasticity." Proceedings of the 11th ACM SIGGRAPH/Eurographics
 conference on Computer Animation. 2012. */
template <typename T>
class StvkHenckyVonMisesModel final
    : public ConstitutiveModel<StvkHenckyVonMisesModel<T>,
                               StvkHenckyVonMisesModelTraits<T>> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(StvkHenckyVonMisesModel);

  using Traits = StvkHenckyVonMisesModelTraits<T>;
  using Data = typename Traits::Data;

  /* Constructs a StvkHenckyVonMisesModel constitutive model with the
   prescribed Young's modulus and Poisson's ratio.
   @param youngs_modulus  Young's modulus of the model, with units of N/m².
   @param poissons_ratio  Poisson's ratio of the model, unitless.
   @param yield_stress    Yield stress for plasticitiy, N/m². We treat the model
                          as elastic if yield_stress is negative.
   @pre youngs_modulus >= 0.
   @pre -1 < poissons_ratio < 0.5. */
  StvkHenckyVonMisesModel(const T& youngs_modulus, const T& poissons_ratio,
                          const T& yield_stress);

  const T& youngs_modulus() const { return E_; }

  const T& poissons_ratio() const { return nu_; }

  /* Returns the shear modulus (Lamé's second parameter) which is given by
   `E/(2*(1+nu))` where `E` is the Young's modulus and `nu` is the Poisson's
   ratio. See `fem::internal::CalcLameParameters()`. */
  const T& shear_modulus() const { return mu_; }

  /* Returns the Lamé's first parameter which is given by
   `E*nu/((1+nu)*(1-2*nu))` where `E` is the Young's modulus and `nu` is the
   Poisson's ratio. See `fem::internal::CalcLameParameters()`. */
  const T& lame_first_parameter() const { return lambda_; }

 private:
  friend ConstitutiveModel<StvkHenckyVonMisesModel<T>,
                           StvkHenckyVonMisesModelTraits<T>>;

  /* Shadows ConstitutiveModel::CalcElasticEnergyDensityImpl() as required by
   the CRTP base class. */
  void CalcElasticEnergyDensityImpl(const Data& data, T* Psi) const;

  /* Shadows ConstitutiveModel::CalcFirstPiolaStressImpl() as required by the
   CRTP base class. */
  void CalcFirstPiolaStressImpl(const Data& data, Matrix3<T>* P) const;

  /* Shadows ConstitutiveModel::CalcFirstPiolaStressDerivativeImpl() as required
   by the CRTP base class. */
  void CalcFirstPiolaStressDerivativeImpl(const Data& data,
                                          Eigen::Matrix<T, 9, 9>* dPdF) const;

  /* Shadows ConstitutiveModel::ProjectStrain() as required by the CRTP base
   class. */
  void ProjectStrainImpl(Matrix3<T>* F, Data* data) const;

  T E_;             // Young's modulus, N/m².
  T nu_;            // Poisson's ratio.
  T mu_;            // Lamé's second parameter/Shear modulus, N/m².
  T lambda_;        // Lamé's first parameter, N/m².
  T yield_stress_;  // Yield stress for plasticitiy, N/m². We treat the model as
                    // elastic if yield_stress_ is negative.
  static constexpr double kSqrt3Over2 = 1.224744871391589;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
