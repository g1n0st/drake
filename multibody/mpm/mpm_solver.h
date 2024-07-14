#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

  int SolveSubsteps(SparseGrid<T>* sparse_grid, 
                    Particles<T>* particles,
                    const mpm::GridData<T>& post_contact_grid_data,
                    const MpmTransfer<T>& transfer,
                    const MpmModel<T>& model, 
                    double dt,
                    MpmSolverScratch<T>* scratch) const {
    if constexpr (!(std::is_same_v<T, double>)) {
      throw;  // only supports double
    }

    mpm::GridData<T> grid_data = post_contact_grid_data;
    double substep_dt = dt / model.substep_count();
    for (int i = 0; i < model.substep_count(); ++i) {
      transfer.G2P(*sparse_grid, grid_data, *particles, &scratch->particles_data, &(scratch->transfer_scratch));

      transfer.UpdateParticlesState(scratch->particles_data, substep_dt, particles);
      
      particles->AdvectParticles(substep_dt);
      
      // NOTE(changyu): the last one P2G will not be counted and will be re-performed in SolveGridVelocities
      transfer.SetUpTransfer(sparse_grid, particles);
      transfer.P2G(*particles, *sparse_grid,
                  &grid_data, &(scratch->transfer_scratch));

      grid_data.ApplyExplicitForceImpulsesToVelocities(substep_dt, model.gravity());
      if (model.newton_params().apply_ground) {
        UpdateCollisionNodesWithGround(*sparse_grid,
                                      &(scratch->collision_nodes));

        grid_data.ProjectionGround(scratch->collision_nodes,
                                              model.newton_params().sticky_ground);
      }
    }

    return model.substep_count();
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

    int count = 0;
    if (model.integrator() == MpmIntegratorType::NewSubstep) {
      count = model.substep_count();
      double substep_dt = dt / count;
      transfer.P2G(mpm_state.particles, mpm_state.sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));
        grid_data_free_motion->ApplyExplicitForceImpulsesToVelocities(substep_dt, model.gravity());
        if (params.apply_ground) {
          UpdateCollisionNodesWithGround(mpm_state.sparse_grid,
                                        &(scratch->collision_nodes));

          grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
        }
      std::cout << "New Substepping " << count << " iterations.\n"
                << "num active nodes: "
                << grid_data_free_motion->num_active_nodes() << std::endl;
    }
    else if (model.integrator() == MpmIntegratorType::Explicit) {
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
    else if (model.integrator() == MpmIntegratorType::OldSubstep) {
      // throw; // NOTE(changyu): This scheme is already proved wrong and deprecated. Will be deleted in future.
      count = model.substep_count();
      double substep_dt = dt / count;

      // Particles<T> temp_initial_particles = mpm_state.particles;
      Particles<T> temp_particles = mpm_state.particles;
      SparseGrid<T> temp_sparse_grid = mpm_state.sparse_grid;

      // std::cout << "Inital Momentum" << mpm_state.particles.ComputeTotalMassMomentum().total_angular_momentum.norm() << std::endl;

      for (int i = 0; i < count; ++i) {
        // transfer.SetUpTransfer(&(temp_sparse_grid), &(temp_particles));
        transfer.P2G(temp_particles, temp_sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));
        // std::cout << "P2G Momentum" << temp_sparse_grid.ComputeTotalMassMomentum(*grid_data_free_motion).total_angular_momentum.norm() << std::endl;

        grid_data_free_motion->ApplyExplicitForceImpulsesToVelocities(substep_dt, model.gravity());
        // std::cout << "Apply Force Momentum" << temp_sparse_grid.ComputeTotalMassMomentum(*grid_data_free_motion).total_angular_momentum.norm() << std::endl;
        if (params.apply_ground) {
          UpdateCollisionNodesWithGround(temp_sparse_grid,
                                        &(scratch->collision_nodes));

          grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
        }

        transfer.G2P(temp_sparse_grid, *grid_data_free_motion, temp_particles, &scratch->particles_data, &(scratch->transfer_scratch));
        temp_particles.SetVelocities(scratch->particles_data.particle_velocites_next);
        temp_particles.SetBMatrices(scratch->particles_data.particle_B_matrices_next);
        temp_particles.UpdateTrialDeformationGradients(substep_dt, scratch->particles_data.particle_grad_v_next);
        temp_particles.UpdateElasticDeformationGradientsAndStresses();

        // std::cout << "G2P Momentum" << temp_particles.ComputeTotalMassMomentum().total_angular_momentum.norm() << std::endl;

        // NOTE(changyu): Advect position here and map velocity field back is incorrect.
        // temp_particles.AdvectParticles(substep_dt);
      }


      // NOTE(changyu): This is merely for finite difference (x*-xn)/dt, so we don't need this anymore.
      /*
      temp_initial_particles.ResetToInitialOrder();
      temp_particles.ResetToInitialOrder();

      for (size_t i = 0; i < temp_initial_particles.num_particles(); ++i) {
        temp_initial_particles.SetVelocityAt(i, temp_particles.GetVelocityAt(i));
        temp_initial_particles.SetBMatrixAt(i, temp_particles.GetBMatrixAt(i));
        // NOTE(changyu): Use v*=(x*-xn)/dt will have lagged effect even under constant graivty.
        // temp_initial_particles.SetVelocityAt(i, (temp_particles.GetPositionAt(i) - temp_initial_particles.GetPositionAt(i)) / dt);
      }

      transfer.SetUpTransfer(&(temp_sparse_grid), &(temp_initial_particles));
      transfer.P2G(temp_initial_particles, temp_sparse_grid,
                    grid_data_free_motion, &(scratch->transfer_scratch));
      std::cout << "P2G(Final) Momentum" << temp_sparse_grid.ComputeTotalMassMomentum(*grid_data_free_motion).total_angular_momentum.norm() << std::endl;
      if (params.apply_ground) {
        UpdateCollisionNodesWithGround(temp_sparse_grid,
                                        &(scratch->collision_nodes));
        grid_data_free_motion->ProjectionGround(scratch->collision_nodes,
                                                params.sticky_ground);
      }
      */
      std::cout << "Substepping " << count << " iterations.\n"
                << "num active nodes: "
                << grid_data_free_motion->num_active_nodes() << std::endl;
    } 
    
    else {
      transfer.P2G(mpm_state.particles, mpm_state.sparse_grid,
                  grid_data_free_motion, &(scratch->transfer_scratch));
      if (params.apply_ground) {
        std::cout << "applying ground" << std::endl;
        UpdateCollisionNodesWithGround(mpm_state.sparse_grid,
                                      &(scratch->collision_nodes));
      }
      count = 0;
      DeformationState<T> deformation_state(
          mpm_state.particles, mpm_state.sparse_grid, *grid_data_free_motion);
      scratch->v_prev = grid_data_free_motion->velocities();

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
    }

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
        (*v)(3 * node_idx + 2) = (*v)(3 * node_idx + 2) <= 0.0 ? 0.0 : (*v)(3 * node_idx + 2);
      }
    }
  }
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
