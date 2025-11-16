#include "drake/multibody/mpm/mpm_model.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MpmModel<T>::ComputeMinusDEnergyDV(
    const MpmTransfer<T>& transfer, const std::vector<Vector3<T>>& v_prev,
    const DeformationState<T>& deformation_state, double dt,
    Eigen::VectorX<T>* minus_dedv, TransferScratch<T>* scratch) const {
  DRAKE_ASSERT(minus_dedv != nullptr);
  DRAKE_ASSERT(scratch != nullptr);
  ComputeMinusDElasticEnergyDV(transfer, deformation_state, dt, minus_dedv,
                               scratch);
  MinusDKineticEnergyDVAndDGravitationalEnergyDV(v_prev, deformation_state, dt,
                                                 minus_dedv);
}

template <typename T>
void MpmModel<T>::ComputeMinusDElasticEnergyDV(
    const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    Eigen::VectorX<T>* minus_dedv, TransferScratch<T>* scratch) const {
  // -dedx
  transfer.ComputeGridElasticForces(
      deformation_state.particles(), deformation_state.sparse_grid(),
      deformation_state.Ps(), minus_dedv, scratch);
  // transform to -dedv, differ by a scale of dt due to chain rule
  (*minus_dedv) *= dt;
}

template <typename T>
void MpmModel<T>::ComputeD2EnergyDV2(
    const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    MatrixX<T>* hessian) const {
  unused(dt);
  ComputeD2ElasticEnergyDV2(transfer, deformation_state, 0, hessian);
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    (*hessian)(3 * i, 3 * i) += deformation_state.grid_data().GetMassAt(i);
    (*hessian)(3 * i + 1, 3 * i + 1) +=
        deformation_state.grid_data().GetMassAt(i);
    (*hessian)(3 * i + 2, 3 * i + 2) +=
        deformation_state.grid_data().GetMassAt(i);
  }
}

template <typename T>
void MpmModel<T>::ComputeD2ElasticEnergyDV2(
    const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    MatrixX<T>* hessian) const {
  transfer.ComputeGridElasticHessian(deformation_state.particles(),
                                     deformation_state.sparse_grid(),
                                     deformation_state.dPdFs(), hessian);
  (*hessian) *= (dt * dt);
}

template <typename T>
void MpmModel<T>::AddD2EnergyDV2TimesZ(
    const Eigen::VectorX<T>& z, const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    Eigen::VectorX<T>* result) const {
  transfer.AddD2ElasticEnergyDV2TimesZ(z, deformation_state.particles(),
                                       deformation_state.sparse_grid(),
                                       deformation_state.dPdFs(), dt, result);
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    (*result).segment(3 * i, 3) +=
        deformation_state.grid_data().GetMassAt(i) * z.segment(3 * i, 3);
  }
}

template <typename T>
void MpmModel<T>::AddD2ElasticEnergyDV2TimesZ(
    const Eigen::VectorX<T>& z, const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    Eigen::VectorX<T>* result) const {
  transfer.AddD2ElasticEnergyDV2TimesZ(z, deformation_state.particles(),
                                       deformation_state.sparse_grid(),
                                       deformation_state.dPdFs(), dt, result);
}

template <typename T>
T MpmModel<T>::ComputeKineticAndGravitationalEnergy(
    const std::vector<Vector3<T>>& v_prev,
    const DeformationState<T>& deformation_state, double dt) const {
  T sum = 0;
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    const Vector3<T>& dv =
        deformation_state.grid_data().GetVelocityAt(i) - v_prev[i];
    sum +=
        0.5 * deformation_state.grid_data().GetMassAt(i) * (dv).squaredNorm();
    sum -= dt * deformation_state.grid_data().GetMassAt(i) * dv.dot(gravity_);
  }
  return sum;
}

template <typename T>
void MpmModel<T>::MinusDKineticEnergyDVAndDGravitationalEnergyDV(
    const std::vector<Vector3<T>>& v_prev,
    const DeformationState<T>& deformation_state, double dt,
    Eigen::VectorX<T>* result) const {
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    const Vector3<T>& dv =
        deformation_state.grid_data().GetVelocityAt(i) - v_prev[i];
    (*result).segment(3 * i, 3) -=
        (deformation_state.grid_data().GetMassAt(i) * dv -
         dt * deformation_state.grid_data().GetMassAt(i) * gravity_);
  }
}

template class MpmModel<double>;
template class MpmModel<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
