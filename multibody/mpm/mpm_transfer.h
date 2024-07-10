#pragma once

#include <array>
#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_3x3_sparse_matrix.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/mpm_state.h"
#include "drake/multibody/mpm/pad.h"
#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/particles_data.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * An implementation of MPM's transfer schemes. We follow Section 10.5 in
 * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
 */
template <typename T>
class MpmTransfer {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmTransfer);

  MpmTransfer() {}

  /**
   * Given current configuration of grid and particles, performs all necessary
   * preparations for transferring, including sorting and computing weights.
   */
  void SetUpTransfer(SparseGrid<T>* grid, Particles<T>* particles) const;

  /**
   * Particles to grid transfer.
   * See Section 10.1 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given particle data, writes to grid_data.
   * @note grid_data is first cleared before "transferring" particle data to
   * grid_data.
   */
  void P2G(const Particles<T>& particles, const SparseGrid<T>& grid,
           GridData<T>* grid_data, TransferScratch<T>* scratch) const;

  /**
   * Grid to particles transfer.
   * See Section 10.1 and 10.2 in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * Given grid_data, writes particle v, B, and grad_v to particles_data.
   */
  void G2P(const SparseGrid<T>& grid, const GridData<T>& grid_data,
           const Particles<T>& particles, ParticlesData<T>* particles_data,
           TransferScratch<T>* scratch) const;

  /**
   * Updates velocity, B-matrix, trial deformation gradient, elastic deformation
   * gradient, and stress P for all particles.
   * @note Return mapping and constitutive model are invoked to get elastic
   * deformation gradient and stress.
   * @note This updates everything except for particle positions.
   */
  void UpdateParticlesState(const ParticlesData<T>& particles_data, double dt,
                            Particles<T>* particles) const;

  /**
   * Computes the grid_forces at all active grid nodes.
   * fᵢ = − ∑ₚ V⁰ₚ P F₀ᵀ ∇ wᵢₚ
   * Also see the accompanied energy_derivatives.md.
   * @note the computation is similar to P2G, with only grid forces being
   * computed.
   * @note the computation depends implicitly on the grid velocities through
   * the first Piola-Kirchhoff stresses `Ps`, which should be computed from the
   * grid velocities.
   * @note the computation depends on the current elastic_deformation_gradient
   * F₀ stored in particles (for chain rule).
   * @pre Ps.size() == particles.num_particles().
   * @note fᵢₐ = grid_elastic_forces(3*i+a).
   */
  void ComputeGridElasticForces(const Particles<T>& particles,
                                const SparseGrid<T>& grid,
                                const std::vector<Matrix3<T>>& Ps,
                                Eigen::VectorX<T>* grid_elastic_forces,
                                TransferScratch<T>* scratch) const {
    particles.SplatStressToP2gPads(Ps, &(scratch->p2g_pads));
    grid.GatherForceFromP2gPads(scratch->p2g_pads, grid_elastic_forces);
  }

  /**
   * Computes the second order derivative of elastic energy w.r.t. grid
   * positions. See the accompanied energy_derivatives.md.
   */
  void ComputeGridElasticHessian(
      const Particles<T>& particles, const SparseGrid<T>& grid,
      const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs,
      MatrixX<T>* hessian) const;

  // compute grid elastic hessian, but a block sparse format
  multibody::contact_solvers::internal::
      BlockSparseLowerTriangularOrSymmetricMatrix<Matrix3<double>, true>
      ComputeGridDElasticEnergyDV2SparseBlockSymmetric(
          const Particles<T>& particles, const SparseGrid<T>& grid,
          const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs, double dt) const {
    if constexpr (std::is_same_v<T, double>) {
      std::vector<Eigen::Matrix<T, 9, 9>> dPdF_contractF0_contractF0 =
          particles.ComputeDPDFContractF0ContractF0(dPdFs);
      std::vector<std::vector<int>> neighbors =
          grid.CalcGridHessianSparsityPattern();
      std::vector<int> block_sizes(grid.num_active_nodes(), 3);
      multibody::contact_solvers::internal::BlockSparsityPattern
          hessian_pattern(std::move(block_sizes), std::move(neighbors));
      multibody::contact_solvers::internal::
          BlockSparseLowerTriangularOrSymmetricMatrix<Matrix3<double>, true>
              hess_sparse(std::move(hessian_pattern));

      Vector3<int> batch_index_3d;

      const std::vector<Vector3<int>>& base_nodes = particles.base_nodes();
      const std::vector<size_t>& batch_starts = particles.batch_starts();
      MatrixX<T> pad_hessian;
      for (size_t i = 0; i < particles.num_batches(); ++i) {
        particles.ComputePadHessianForOneBatch(i, dPdF_contractF0_contractF0,
                                               &pad_hessian);
        AddPadHessianToHessianSymmetricBlockSparse(
            base_nodes[batch_starts[i]], grid, pad_hessian * (dt * dt),
            &hess_sparse);
      }

      return hess_sparse;
    } else {
      throw;
    }
  }

  void AddPadHessianToHessianSymmetricBlockSparse(
      const Vector3<int> batch_index_3d, const SparseGrid<T>& grid,
      const MatrixX<T>& pad_hessian,
      multibody::contact_solvers::internal::
          BlockSparseLowerTriangularOrSymmetricMatrix<Matrix3<double>, true>*
              result) const {
    if constexpr (std::is_same_v<T, double>) {
      // temporary variables
      int idx_local_i, idx_local_j;
      Vector3<int> node_i_idx_3d, node_j_idx_3d;

      for (int ia = -1; ia <= 1; ++ia) {
        for (int ib = -1; ib <= 1; ++ib) {
          for (int ic = -1; ic <= 1; ++ic) {
            idx_local_i = 9 * (ia + 1) + 3 * (ib + 1) + (ic + 1);
            // local index 0-26
            node_i_idx_3d = batch_index_3d + Vector3<int>(ia, ib, ic);
            // global 3d index of node i

            for (int ja = -1; ja <= 1; ++ja) {
              for (int jb = -1; jb <= 1; ++jb) {
                for (int jc = -1; jc <= 1; ++jc) {
                  idx_local_j = 9 * (ja + 1) + 3 * (jb + 1) + (jc + 1);
                  // local index 0-26
                  node_j_idx_3d = batch_index_3d + Vector3<int>(ja, jb, jc);
                  // since symmetric, only add lower triangle
                  if (grid.To1DIndex(node_i_idx_3d) >=
                      grid.To1DIndex(node_j_idx_3d)) {
                    const auto& dfi_dvj_block =
                        pad_hessian.template block<3, 3>(idx_local_i * 3,
                                                         idx_local_j * 3);
                    result->AddToBlock(grid.To1DIndex(node_i_idx_3d),
                                       grid.To1DIndex(node_j_idx_3d),
                                       dfi_dvj_block);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Given m grid node velocities (3mby1) and a particle with index
  // index_particle, compute J (3by3m) such that velocity of the particle = J *
  // grid_velocity
  // tested
  void CalcJacobianGridVToParticleVAtParticle(size_t p,
                                              const Particles<T>& particles,
                                              const SparseGrid<T>& sparse_grid,
                                              MatrixX<T>* J) const {
    DRAKE_DEMAND(!particles.NeedReordering());
    (*J) = MatrixX<T>::Zero(3, 3 * sparse_grid.num_active_nodes());
    const Vector3<int> base_node_p = particles.GetBaseNodeAt(p);
    Matrix3<T> local_J;
    int idx_local;
    int idx_global;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          idx_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
          local_J =
              particles.GetWeightAt(p, idx_local) * Matrix3<T>::Identity();
          idx_global =
              sparse_grid.To1DIndex(base_node_p + Vector3<int>(a, b, c));
          J->template block<3, 3>(0, idx_global * 3) = local_J;
        }
      }
    }
  }

  contact_solvers::internal::Block3x3SparseMatrix<T>
  CalcRTimesJacobianGridVToParticleVWithPermutationAtParticle(
      const Matrix3<T>& R, size_t p, const MpmState<T>& state,
      const MpmGridNodesPermutation<T>& perm) const {
    DRAKE_DEMAND(!state.particles.NeedReordering());

    contact_solvers::internal::Block3x3SparseMatrix<T> J(
        1, perm.nodes_in_contact.size());
    std::vector<std::tuple<int, int, Matrix3<T>>> triplets;

    const Vector3<int> base_node_p = state.particles.GetBaseNodeAt(p);
    int idx_local;
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          idx_local = (c + 1) + 3 * (b + 1) + 9 * (a + 1);
          Matrix3<T> local_J = R * state.particles.GetWeightAt(p, idx_local) *
                               Matrix3<T>::Identity();

          int idx_global =
              state.sparse_grid.To1DIndex(base_node_p + Vector3<int>(a, b, c));
          int idx_global_postperm = perm.permutation.domain_index(idx_global);
          DRAKE_DEMAND(idx_global_postperm <
                       static_cast<int>(perm.nodes_in_contact.size()));

          triplets.emplace_back(std::tuple<int, int, Matrix3<T>>(
              0, idx_global_postperm, std::move(local_J)));
        }
      }
    }
    J.SetFromTriplets(triplets);
    return J;
  }

  /**
   * Computes result += d2(elastic_energy)/dv2 * z.
   * @note d2(elastic_energy)/dv2 = ComputeGridElasticHessian() * dt * dt.
   */
  void AddD2ElasticEnergyDV2TimesZ(
      const Eigen::VectorX<T>& z, const Particles<T>& particles,
      const SparseGrid<T>& grid,
      const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs, double dt,
      Eigen::VectorX<T>* result) const;

 private:
  // Adds the batch_index_3d-th pad hessian (of elastic energy w.r.t grid
  // positions) to the hessian.
  void AddPadHessianToHessian(const Vector3<int> batch_index_3d,
                              const SparseGrid<T>& grid,
                              const MatrixX<T>& pad_hessian,
                              MatrixX<T>* hessian) const;

  // Computes the matrix A in the computation of ElasticHessianTimesZ, for
  // particle p. See energy_derivatives.md for details.
  void ComputeAp(size_t p, const Eigen::VectorX<T>& z,
                 const Particles<T>& particles, const SparseGrid<T>& grid,
                 const Eigen::Matrix<T, 9, 9>& dPdF, Matrix3<T>* Ap) const;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
