#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fstream>
#include <iomanip>
#include <filesystem>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/conjugate_gradient.h"
#include "drake/multibody/mpm/mpm_model.h"
#include "drake/multibody/mpm/mpm_state.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmSolver {
 public:
  MpmSolver() {}

  void ComputeGridDataPrevStep(const MpmState<T>& mpm_state,
                               const MpmTransfer<T>& transfer,
                               mpm::GridData<T>* grid_data_prev_step,
                               MpmSolverScratch<T>* scratch) const {
    if constexpr (!(std::is_same_v<T, double>)) {
      throw;  // only supports double
    }
    transfer.P2G(mpm_state.particles, mpm_state.sparse_grid,
                 grid_data_prev_step, &(scratch->transfer_scratch));
  }

  int SolveGridVelocities(const NewtonParams& params,
                          const MpmState<T>& mpm_state,
                          const MpmTransfer<T>& transfer,
                          const MpmModel<T>& model, double dt,
                          mpm::GridData<T>* grid_data_free_motion,
                          MpmSolverScratch<T>* scratch) const {
    if constexpr (!(std::is_same_v<T, double>)) {
      throw;  // only supports double
    }

    transfer.P2G(mpm_state.particles, mpm_state.sparse_grid,
                  grid_data_free_motion, &(scratch->transfer_scratch));
    scratch->v_prev = grid_data_free_motion->velocities();
    if (params.apply_ground) {
        std::cout << "applying ground" << std::endl;
        UpdateCollisionNodesWithGround(mpm_state.sparse_grid,
                                      &(scratch->collision_nodes));
      }
    int count = 0;
    if (model.integrator() == MpmIntegratorType::Explicit) {
        transfer.P2G(mpm_state.particles, mpm_state.sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));
        grid_data_free_motion->ApplyExplicitForceImpulsesToVelocities(dt, model.gravity());
        if (params.apply_ground) {
          UpdateCollisionNodesWithGround(mpm_state.sparse_grid,
                                        &(scratch->collision_nodes));

          grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
        }
        std::cout << "Single Stage \n"
                << "num active nodes: "
                << grid_data_free_motion->num_active_nodes() << std::endl;
    }
    else if (model.integrator() == MpmIntegratorType::Substep) {
      count = model.substep_count();
      double substep_dt = dt / count;

      SparseGrid<T> temp_sparse_grid = mpm_state.sparse_grid;
      Particles<T> temp_initial_particles = mpm_state.particles;
      Particles<T> temp_particles = mpm_state.particles;

      for (int i = 0; i <  count; ++i) {
        transfer.SetUpTransfer(&(temp_sparse_grid), &(temp_particles));
        transfer.P2G(temp_particles, temp_sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));

        grid_data_free_motion->ApplyExplicitForceImpulsesToVelocities(substep_dt, model.gravity());
        if (params.apply_ground) {
          UpdateCollisionNodesWithGround(temp_sparse_grid,
                                        &(scratch->collision_nodes));

          grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
        }

        transfer.G2P(temp_sparse_grid, *grid_data_free_motion, temp_particles, &scratch->particles_data, &(scratch->transfer_scratch));
        transfer.UpdateParticlesState(scratch->particles_data, substep_dt, &temp_particles);

        // NOTE(changyu): Advect position here and map velocity field back is incorrect.
        temp_particles.AdvectParticles(substep_dt);
      }

      temp_initial_particles.ResetToInitialOrder();
      temp_particles.ResetToInitialOrder();

      for (size_t i = 0; i < temp_initial_particles.num_particles(); ++i) {
        // temp_initial_particles.SetVelocityAt(i, temp_particles.GetVelocityAt(i));
        // NOTE(changyu): Use v*=(x*-xn)/dt will have lagged effect even under constant graivty.
        temp_initial_particles.SetVelocityAt(i, (temp_particles.GetPositionAt(i) - temp_initial_particles.GetPositionAt(i)) / dt);

        // Secant ∇v over the big step:
        // From F^{n+1} ≈ (I + Δt ∇v) F^n  ⇒  ∇v ≈ (F^{n+1} (F^n)^{-1} - I) / Δt
        T h = temp_sparse_grid.h();
        const T D = T(1./4.) * h * h;
        temp_initial_particles.SetBMatrixAt(i, ((temp_particles.GetElasticDeformationGradientAt(i) * temp_initial_particles.GetElasticDeformationGradientAt(i).inverse()) - Matrix3<T>::Identity()) / dt * D);
      }

      transfer.SetUpTransfer(&(temp_sparse_grid), &(temp_initial_particles));
      transfer.P2G(temp_initial_particles, temp_sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));
      if (params.apply_ground) {
        UpdateCollisionNodesWithGround(temp_sparse_grid,
                                        &(scratch->collision_nodes));
        grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
      }
      std::cout << "Substepping " << count << " iterations.\n"
                << "num active nodes: "
                << grid_data_free_motion->num_active_nodes() << std::endl;
    } 
    
    else {
      count = 0;
      DeformationState<T> deformation_state(
          mpm_state.particles, mpm_state.sparse_grid, *grid_data_free_motion);

      for (; count < params.max_newton_iter; ++count) {
        deformation_state.Update(transfer, dt, scratch,
                                (!params.linear_constitutive_model));
        // find minus_gradient
        model.ComputeMinusDEnergyDV(transfer, scratch->v_prev, deformation_state,
                                    dt, &(scratch->minus_dEdv),
                                    &(scratch->transfer_scratch));

        // if (params.apply_ground) {
        //   ProjectCollisionGround(scratch->collision_nodes,
        //   params.sticky_ground,
        //                          &(scratch->minus_dEdv));
        // }
        double gradient_norm = scratch->minus_dEdv.norm();
        if ((gradient_norm < params.newton_gradient_epsilon) && (count > 0))
          break;

        // find dG_ = hessian^-1 * minus_gradient, using CG

        if (params.matrix_free) {
          ConjugateGradient cg;
          if (params.linear_constitutive_model) {
            // if model is linear, cg only needs to be this much accurate for
            // newton to converge in one step
            cg.SetRelativeTolerance(0.5 * params.newton_gradient_epsilon /
                                    std::max(gradient_norm, 1e-6));
          }
          HessianWrapper hessian_wrapper(transfer, model, deformation_state, dt);
          cg.Solve(hessian_wrapper, scratch->minus_dEdv, &(scratch->dG));

        } else {
          // not matrix free, use eigen dense matrix
          Eigen::ConjugateGradient<MatrixX<T>, Eigen::Lower | Eigen::Upper>
              cg_dense;
          if (params.linear_constitutive_model) {
            if (count > 2) {
              throw std::logic_error("linear solver newton does not converge");
            }
            // if model is linear, cg only needs to be this much accurate for
            // newton to converge in one step
            cg_dense.setTolerance(0.5 * params.newton_gradient_epsilon /
                                  std::max(gradient_norm, 1e-6));
          }
          model.ComputeD2EnergyDV2(transfer, deformation_state, dt,
                                  &(scratch->d2Edv2));
          cg_dense.compute(scratch->d2Edv2);
          scratch->dG = cg_dense.solve(scratch->minus_dEdv);
        }

        grid_data_free_motion->AddDG(scratch->dG);
      }
      std::cout << "Newton converged after " << count << " iterations.\n"
                << "num active nodes: "
                << grid_data_free_motion->num_active_nodes() << std::endl;
      if (params.apply_ground) {
        grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
      }

      const int G2P2G_smooth = 0;
      for (int i = 0; i < G2P2G_smooth; ++i) {
        SparseGrid<T> temp_sparse_grid = mpm_state.sparse_grid;
        Particles<T> temp_initial_particles = mpm_state.particles;
        Particles<T> temp_particles = mpm_state.particles;

        transfer.SetUpTransfer(&(temp_sparse_grid), &(temp_particles));
        transfer.G2P(temp_sparse_grid, *grid_data_free_motion, temp_particles, &scratch->particles_data, &(scratch->transfer_scratch));
        transfer.UpdateParticlesStateVOnly(scratch->particles_data, &temp_particles);
        transfer.P2G(temp_particles, temp_sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));

      }
    }

    // ----------------------------------------------------------------------
    // Residual-aware re-anchoring of free-motion velocity:
    // v† = v* - A^{-1} r, where r = m(v*), A = ∂m/∂v|_{v*} = d²E/dv².
    // This makes the free-motion anchor consistent for the post-contact SAP.
    // Only apply to implicit integrator.
    // ----------------------------------------------------------------------
    if (false) {
      // Build state at current v* (grid_data_free_motion holds v* now).
      DeformationState<T> def_vstar(
          mpm_state.particles, mpm_state.sparse_grid, *grid_data_free_motion);
      def_vstar.Update(transfer, dt, scratch,
                      (!params.linear_constitutive_model));

      // r = m(v*) = - ( -dE/dv ) under our discretization.
      model.ComputeMinusDEnergyDV(transfer, scratch->v_prev, def_vstar, dt,
                                  &(scratch->minus_dEdv),
                                  &(scratch->transfer_scratch));
      Eigen::VectorX<T> r = -scratch->minus_dEdv;

      // A = d²E/dv² (SPD candidate).
      model.ComputeD2EnergyDV2(transfer, def_vstar, dt, &(scratch->d2Edv2));
      const MatrixX<T>& A = scratch->d2Edv2;

      // Solve A w = r via LDLT; add tiny Tikhonov if needed.
      Eigen::LDLT<MatrixX<T>> ldlt(A);
      if (ldlt.info() != Eigen::Success) {
        MatrixX<T> Areg = A;
        Areg.diagonal().array() += static_cast<T>(1e-12);
        ldlt.compute(Areg);
      }
      Eigen::VectorX<T> w = ldlt.solve(r);

      // v† = v* - w
      grid_data_free_motion->AddDG(-w);

      // Re-apply ground projection if requested, to keep constraints satisfied.
      if (params.apply_ground) {
        grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
      }
    }

    // NOTE (changyu): compute final residual
    DeformationState<T> deformation_state_v_star(
    mpm_state.particles, mpm_state.sparse_grid, *grid_data_free_motion);
    // scratch->v_prev = grid_data_free_motion->velocities(); NOTE: v^n, do not modify it!
    deformation_state_v_star.Update(transfer, dt, scratch,
                              (!params.linear_constitutive_model));
    // find minus_gradient
    model.ComputeMinusDEnergyDV(transfer, scratch->v_prev, deformation_state_v_star,
                                dt, &(scratch->minus_dEdv),
                                &(scratch->transfer_scratch));
    model.ComputeD2EnergyDV2(transfer, deformation_state_v_star, dt,
                                  &(scratch->d2Edv2));
    // In our discretization: m(v*) = -minus_dEdv, and A = d m / d v |_{v*} = d2Edv2.
    Eigen::VectorX<T> m_free = -scratch->minus_dEdv;   // m(v*), momentum residual
    const MatrixX<T>& A = scratch->d2Edv2;             // SPD candidate

    // Residual norms: Euclidean and A^{-1}-norm (natural for SAP).
    const T res_l2 = m_free.norm();
    const T res_linf = m_free.cwiseAbs().maxCoeff();

    // Compute ||m(v*)||_{A^{-1}} robustly via LDLT with tiny Tikhonov if needed.
    auto Ainv_norm = [](const MatrixX<T>& A_local,
                        const Eigen::Ref<const VectorX<T>>& r) -> T {
      Eigen::LDLT<MatrixX<T>> ldlt(A_local);
      if (ldlt.info() != Eigen::Success) {
        MatrixX<T> Areg = A_local;
        // Tikhonov regularization to handle near-singular cases.
        using Scalar = typename MatrixX<T>::Scalar;
        const Scalar eps = static_cast<Scalar>(1e-12);
        Areg.diagonal().array() += eps;
        ldlt.compute(Areg);
      }
      VectorX<T> z = ldlt.solve(r);
      return std::sqrt(std::max<typename VectorX<T>::Scalar>(r.dot(z), 0.0));
    };

    const T res_Ainv = Ainv_norm(A, m_free);
    std::cout << "\033[31m"
              << "[SAP] Free-motion residual ||m(v*)||_2 = " << res_l2
              << ", ||m(v*)||_inf = " << res_linf
              << ", ||m(v*)||_{A^{-1}} = " << res_Ainv
              << "\033[0m" << std::endl;
    
    // Append one CSV row: iter,res_l2,res_linf,res_Ainv  (with header once)
    auto append_residual_row = [](const std::string& path,
                                  int iter, double r2, double rinf, double rAinv) {
      namespace fs = std::filesystem;
      const bool need_header = !fs::exists(path) || fs::file_size(path) == 0;

      std::ofstream ofs(path, std::ios::app);
      ofs.setf(std::ios::fixed);
      ofs << std::setprecision(16);
      if (need_header) ofs << "iter,res_l2,res_linf,res_Ainv\n";
      ofs << iter << "," << r2 << "," << rinf << "," << rAinv << "\n";
    };

    append_residual_row("residuals.txt",
                      /*iter=*/count,
                      static_cast<double>(res_l2),
                      static_cast<double>(res_linf),
                      static_cast<double>(res_Ainv));

    return count;
  }

 private:
  void UpdateCollisionNodesWithGround(
      const SparseGrid<T>& sparse_grid,
      std::vector<size_t>* collision_nodes) const {
    collision_nodes->clear();
    for (size_t i = 0; i < sparse_grid.num_active_nodes(); ++i) {
      if (sparse_grid.To3DIndex(i)(2) <= 0) {
        collision_nodes->push_back(i);
      }
    }
  }

  void ProjectCollisionGround(const std::vector<size_t>& collision_nodes,
                              bool sticky_ground, Eigen::VectorX<T>* v) const {
    for (auto node_idx : collision_nodes) {
      if (sticky_ground) {
        (*v).segment(3 * node_idx, 3) = Vector3<T>(0, 0, 0);
      } else {
        (*v)(3 * node_idx + 2) = 0.0;
      }
    }
  }
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
