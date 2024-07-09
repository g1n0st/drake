#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/math/rigid_transform.h"
#include "drake/multibody/mpm/internal/analytic_level_set.h"
#include "drake/multibody/mpm/mpm_transfer.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"


namespace drake {
namespace multibody {
namespace mpm {

/**
 * Advance from tₙ to tₙ₊₁ using implicit scheme:
 *
 * Have the following objects in hands
 * Particles<double> particles; -- carries state of Lagrangian particles at tₙ,
 * this is the only meaningful data at this moment
 *
 * SparseGrid<double> sparse_grid; -- auxiliary structure for indexing
 *
 * GridData<double> grid_data; -- zero-initialized
 * ParticlesData<double> particles_data; -- zero-initialized
 *
 * MpmModel<double> mpm_model; -- compute energy and its derivatives
 *
 * MpmTransfer<double> mpm_transfer; -- transfer functions
 * TransferScratch<double> scratch; -- scratch data for transfer
 *
 * dG = ∞
 * mpm_transfer.SetUpTransfer(&sparse_grid, &particles);
 * mpm_transfer.P2G(particles, sparse_grid, &grid_data, &transfer_scratch);
 *
 * DeformationState<double> state(particles, sparse_grid, grid_data);
 * while |dG|>ε:
 *    state.Update(mpm_transfer, dt, &scratch);
 *    energy = mpm_model.ComputeEnergy(state);
 *    force = mpm_model.ComputeForce(state, mpm_transfer, scratch);
 *    hessian = mpm_model.ComputeHessian(state, mpm_transfer);
 *    dG = LinearSolve(force, energy);
 *    grid_data += dG
 *
 * particles at tₙ₊₁ = mpm_transfer.G2P(grid_data, dt) and
 * mpm_transfer.UpdateParticlesState(), updating v, C, F, P, x, to be stated
 * more clearly
 *
 * TODO(zeshunzong): write an explicit scheme
 */

struct NewtonParams {
  int max_newton_iter = 500;
  double newton_gradient_epsilon = 1e-5;
  bool matrix_free = true;
  bool linear_constitutive_model = true;
  bool apply_ground = false;
  bool sticky_ground = false;
};

template <typename T>
class DeformationState {
  /**
   * A temporary holder for data needed in Newton iterations.
   */
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformationState);
  DeformationState(const Particles<T>& particles,
                   const SparseGrid<T>& sparse_grid,
                   // NOLINTNEXTLINE
                   GridData<T>& grid_data)
      : particles_(particles),
        sparse_grid_(sparse_grid),
        grid_data_(grid_data) {
    Fs_.resize(particles.num_particles());
    Ps_.resize(particles.num_particles());
    dPdFs_.resize(particles.num_particles());
  }

  /**
   * Computes F, P and dPdF for each particle, from the grid_data
   * @pre !particles.NeedReordering()
   * @pre sparse_grid is compatible with current particles
   */
  void Update(const MpmTransfer<T>& transfer, double dt,
              MpmSolverScratch<T>* scratch, bool project_pd = false) {
    // TODO(zeshunzong): some other data is computed but not used in
    // particles_data
    transfer.G2P(sparse_grid_, grid_data_, particles_,
                 &(scratch->particles_data), &(scratch->transfer_scratch));
    particles_.ComputeFsPsdPdFs(scratch->particles_data.particle_grad_v_next,
                                dt, &Fs_, &Ps_, &dPdFs_, project_pd);
  }

  const std::vector<Matrix3<T>>& Fs() const { return Fs_; }
  const std::vector<Matrix3<T>>& Ps() const { return Ps_; }
  const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs() const { return dPdFs_; }

  const Particles<T>& particles() const { return particles_; }
  const SparseGrid<T>& sparse_grid() const { return sparse_grid_; }
  const GridData<T>& grid_data() const { return grid_data_; }

 private:
  std::vector<Matrix3<T>> Fs_{};
  std::vector<Matrix3<T>> Ps_{};
  std::vector<Eigen::Matrix<T, 9, 9>> dPdFs_{};

  const Particles<T>& particles_;
  const SparseGrid<T>& sparse_grid_;
  GridData<T>& grid_data_;
};

template <typename T>
struct MpmInitialObjectParameters {
  std::unique_ptr<internal::AnalyticLevelSet> level_set;
  std::unique_ptr<constitutive_model::ElastoPlasticModel<T>> constitutive_model;
  std::unique_ptr<math::RigidTransform<T>> pose;
  double density;
  double grid_h;

  MpmInitialObjectParameters(
      std::unique_ptr<internal::AnalyticLevelSet> level_set_in,
      std::unique_ptr<constitutive_model::ElastoPlasticModel<T>>
          constitutive_model_in,
      std::unique_ptr<math::RigidTransform<T>> pose_in, double density_in,
      double h_in)
      : density(density_in), grid_h(h_in) {
    level_set = std::move(level_set_in);
    constitutive_model = std::move(constitutive_model_in);
    pose = std::move(pose_in);
  }
};

template <typename T>
class MpmModel {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmModel);

  MpmModel() {}

  void ApplyMpmGround() { newton_params_.apply_ground = true; }

  void StoreInitialObjectParams(
      std::unique_ptr<internal::AnalyticLevelSet> level_set_in,
      std::unique_ptr<constitutive_model::ElastoPlasticModel<T>>
          constitutive_model_in,
      std::unique_ptr<math::RigidTransform<T>> pose_in, double density_in,
      double h_in) {
    initial_object_params_ = std::make_unique<MpmInitialObjectParameters<T>>(
        std::move(level_set_in), std::move(constitutive_model_in),
        std::move(pose_in), density_in, h_in);

    if (initial_object_params_->constitutive_model->IsLinearModel()) {
      newton_params_.linear_constitutive_model = true;
    }
    else {
      newton_params_.linear_constitutive_model = false;
    }
  }

  const MpmInitialObjectParameters<T>& InitialObjectParams() const {
    DRAKE_DEMAND(initial_object_params_ != nullptr);
    return *initial_object_params_;
  }

  void SetMpmStateIndex(const systems::AbstractStateIndex& index) {
    mpm_state_index_ = index;
  }

  const systems::AbstractStateIndex& mpm_state_index() const {
    return mpm_state_index_;
  }

  /**
   * Total energy = elastic energy + kinetic energy + gravitational energy.
   */
  T ComputeEnergy(const std::vector<Vector3<T>>& v_prev,
                  const DeformationState<T>& deformation_state,
                  double dt) const {
    return ComputeElasticEnergy(deformation_state) +
           ComputeKineticAndGravitationalEnergy(v_prev, deformation_state, dt);
  }

  /**
   * elastic energy = ∑ₚ V⁰ₚ ψ(Fₚ), where Fₚ depends on grid velocities
   */
  T ComputeElasticEnergy(const DeformationState<T>& deformation_state) const {
    return deformation_state.particles().ComputeTotalElasticEnergy(
        deformation_state.Fs());
  }

  /**
   * Computes - d(total_energy)/dv = -d(elastic_energy)/dv -
   * d(kinetic_energy)/dv - d(gravitational_energy)/dv.
   */
  void ComputeMinusDEnergyDV(const MpmTransfer<T>& transfer,
                             const std::vector<Vector3<T>>& v_prev,
                             const DeformationState<T>& deformation_state,
                             double dt, Eigen::VectorX<T>* minus_dedv,
                             TransferScratch<T>* scratch) const;

  /**
   * Computes -d(elastic_energy)/dv.
   */
  void ComputeMinusDElasticEnergyDV(
      const MpmTransfer<T>& transfer,
      const DeformationState<T>& deformation_state, double dt,
      Eigen::VectorX<T>* dEnergydV, TransferScratch<T>* scratch) const;

  /**
   * Computes the 3*num_active_nodes()by3*num_active_nodes() hessian matrix d^2
   * (total_energy) / dv^2.
   */
  void ComputeD2EnergyDV2(const MpmTransfer<T>& transfer,
                          const DeformationState<T>& deformation_state,
                          double dt, MatrixX<T>* hessian) const;

  /**
   * Computes the 3*num_active_nodes()by3*num_active_nodes() hessian matrix d^2
   * (elastic_energy) / dv^2.
   */
  void ComputeD2ElasticEnergyDV2(const MpmTransfer<T>& transfer,
                                 const DeformationState<T>& deformation_state,
                                 double dt, MatrixX<T>* hessian) const;

  /**
   * d total_energy dv^2, in a BlockSparseLowerTriangularOrSymmetricMatrix form
   */
  multibody::contact_solvers::internal::
      BlockSparseLowerTriangularOrSymmetricMatrix<Matrix3<double>, true>
      ComputeD2EnergyDV2SymmetricBlockSparse(
          const MpmTransfer<T>& transfer,
          const DeformationState<T>& deformation_state, double dt) const {
    if constexpr (std::is_same_v<T, double>) {
      unused(dt);
      multibody::contact_solvers::internal::
          BlockSparseLowerTriangularOrSymmetricMatrix<Matrix3<double>, true>
              result =
                  transfer.ComputeGridDElasticEnergyDV2SparseBlockSymmetric(
                      deformation_state.particles(),
                      deformation_state.sparse_grid(),
                      deformation_state.dPdFs(), 0.0);
      // add the mass part
      for (size_t i = 0; i < deformation_state.sparse_grid().num_active_nodes();
           ++i) {
        Eigen::Matrix3<double> diagonal_mass = Eigen::Matrix3<double>::Zero();
        diagonal_mass.diagonal().setConstant(
            deformation_state.grid_data().GetMassAt(i));
        result.AddToBlock(i, i, diagonal_mass);
      }

      return result;
    } else {
      throw;
    }
  }

  /**
   * Computes result += ComputeD2EnergyDV2() * z.
   */
  void AddD2EnergyDV2TimesZ(const Eigen::VectorX<T>& z,
                            const MpmTransfer<T>& transfer,
                            const DeformationState<T>& deformation_state,
                            double dt, Eigen::VectorX<T>* result) const;

  /**
   * Computes result += ComputeD2ElasticEnergyDV2() * z.
   */
  void AddD2ElasticEnergyDV2TimesZ(const Eigen::VectorX<T>& z,
                                   const MpmTransfer<T>& transfer,
                                   const DeformationState<T>& deformation_state,
                                   double dt, Eigen::VectorX<T>* result) const;

  void SetGravity(const Vector3<T>& g) { gravity_ = g; }

  const Vector3<T>& gravity() const { return gravity_; }

  // precondition for cg solve, currently this is MASS
  // that is, in absense of elastic deformation, this is equivalent to diagonal
  // preconditioner
  void Precondition(const DeformationState<T>& deformation_state,
                    const Eigen::VectorX<T>& rhs, Eigen::VectorX<T>* x) const {
    for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
         ++i) {
      if (deformation_state.grid_data().masses()[i] > 0.0) {
        (*x).segment(3 * i, 3) =
            rhs.segment(3 * i, 3) / deformation_state.grid_data().masses()[i];
      } else {
        (*x).segment(3 * i, 3) = rhs.segment(3 * i, 3);
      }
    }
  }

  const NewtonParams& newton_params() const { return newton_params_; }

  int min_num_particles_per_cell() const { return min_num_particles_per_cell_; }

  void SetMinNumParticlesPerCell(int x) { min_num_particles_per_cell_ = x; }

  double friction_mu() const { return friction_mu_; }

  void SetFrictionMu(double mu) { friction_mu_ = mu; }

 private:
  // Kinetic energy = 0.5 * m * (v - v_prev)ᵀ(v - v_prev).
  // Gravitational energy = - m*dt*gᵀv. Since we only care about its gradient,
  // we can do - m*dt*gᵀ(v - v_prev).
  T ComputeKineticAndGravitationalEnergy(
      const std::vector<Vector3<T>>& v_prev,
      const DeformationState<T>& deformation_state, double dt) const;

  // -d(kinetic_energy)/dv = - m * (v - v₀).
  // -d(gravitational_energy)/dv = dt * mg.
  // result = result -d(kinetic_energy)/dv -d(gravitational_energy)/dv
  void MinusDKineticEnergyDVAndDGravitationalEnergyDV(
      const std::vector<Vector3<T>>& v_prev,
      const DeformationState<T>& deformation_state, double dt,
      Eigen::VectorX<T>* result) const;

  // Vector3<T> gravity_{2.0, 0.0, -2.0};
  Vector3<T> gravity_{0.0, 0.0, -8.0};

  // consider having a list of those?
  std::unique_ptr<MpmInitialObjectParameters<T>> initial_object_params_;
  // the state index where we store mpm_state inside context
  systems::AbstractStateIndex mpm_state_index_;

  int min_num_particles_per_cell_ = 1;

  NewtonParams newton_params_;

  double friction_mu_ = 0.1;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
