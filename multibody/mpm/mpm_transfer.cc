#include "drake/multibody/mpm/mpm_transfer.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
void MpmTransfer<T>::SetUpTransfer(SparseGrid<T>* grid,
                                   Particles<T>* particles) const {
  particles->Prepare(grid->h());
  grid->MarkActiveNodes(particles->base_nodes());
}

template <typename T>
void MpmTransfer<T>::P2G(const Particles<T>& particles,
                         const SparseGrid<T>& grid, GridData<T>* grid_data,
                         TransferScratch<T>* scratch) const {
  particles.SplatToP2gPads(grid.h(), &(scratch->p2g_pads));
  grid.GatherFromP2gPads(scratch->p2g_pads, grid_data);
}

template <typename T>
void MpmTransfer<T>::G2P(const SparseGrid<T>& grid,
                         const GridData<T>& grid_data,
                         const Particles<T>& particles,
                         ParticlesData<T>* particles_data,
                         TransferScratch<T>* scratch) const {
  DRAKE_DEMAND(!particles.NeedReordering());
  particles_data->Resize(particles.num_particles());
  Vector3<int> idx_3d;
  // loop over all batches
  Vector3<int> batch_idx_3d;
  const std::vector<Vector3<int>>& base_nodes = particles.base_nodes();
  const std::vector<size_t>& batch_starts = particles.batch_starts();
  for (size_t i = 0; i < particles.num_batches(); ++i) {
    batch_idx_3d = base_nodes[batch_starts[i]];

    // form the g2p_pad for this batch
    scratch->g2p_pad.SetZero();
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          idx_3d = Vector3<int>(batch_idx_3d(0) + a, batch_idx_3d(1) + b,
                                batch_idx_3d(2) + c);
          const Vector3<double> position =
              internal::ComputePositionFromIndex3D(idx_3d, grid.h());
          const Vector3<T>& velocity =
              grid_data.GetVelocityAt(grid.To1DIndex(idx_3d));
          scratch->g2p_pad.SetPositionAndVelocityAt(a, b, c, position,
                                                    velocity);
        }
      }
    }
    // write particle v, B, and grad v to particles_data
    particles.WriteParticlesDataFromG2pPad(i, scratch->g2p_pad, particles_data);
  }
}

template <typename T>
void MpmTransfer<T>::UpdateParticlesState(
    const ParticlesData<T>& particles_data, double dt,
    Particles<T>* particles) const {
  particles->SetVelocities(particles_data.particle_velocites_next);
  particles->SetBMatrices(particles_data.particle_B_matrices_next);
  particles->UpdateTrialDeformationGradients(
      dt, particles_data.particle_grad_v_next);
  particles->UpdateElasticDeformationGradientsAndStresses();
}

template <typename T>
void MpmTransfer<T>::ComputeGridElasticHessian(
    const Particles<T>& particles, const SparseGrid<T>& grid,
    const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs,
    MatrixX<T>* hessian) const {
  DRAKE_ASSERT(hessian != nullptr);
  // initialize
  hessian->resize(grid.num_active_nodes() * 3, grid.num_active_nodes() * 3);
  hessian->setZero();

  std::vector<Eigen::Matrix<T, 9, 9>> dPdF_contractF0_contractF0 =
      particles.ComputeDPDFContractF0ContractF0(dPdFs);
  // loop over each batch
  const std::vector<Vector3<int>>& base_nodes = particles.base_nodes();
  const std::vector<size_t>& batch_starts = particles.batch_starts();
  MatrixX<T> pad_hessian;
  
  
  for (size_t i = 0; i < particles.num_batches(); ++i) {
    // compute pad_hessian for this batch
    particles.ComputePadHessianForOneBatch(i, dPdF_contractF0_contractF0,
                                           &pad_hessian);
    // write pad_hessian to global hessian
    AddPadHessianToHessian(base_nodes[batch_starts[i]], grid, pad_hessian,
                           hessian);
  }
}

template <typename T>
void MpmTransfer<T>::AddPadHessianToHessian(const Vector3<int> batch_index_3d,
                                            const SparseGrid<T>& grid,
                                            const MatrixX<T>& pad_hessian,
                                            MatrixX<T>* hessian) const {
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
              // global 3d index of node i
              const auto& dfi_dvj_block = pad_hessian.template block<3, 3>(
                  idx_local_i * 3, idx_local_j * 3);
              // get the local dfi_dvj block
              (*hessian).block(grid.To1DIndex(node_i_idx_3d) * 3,
                               grid.To1DIndex(node_j_idx_3d) * 3, 3, 3) +=
                  dfi_dvj_block;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void MpmTransfer<T>::AddD2ElasticEnergyDV2TimesZ(
    const Eigen::VectorX<T>& z, const Particles<T>& particles,
    const SparseGrid<T>& grid, const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs,
    double dt, Eigen::VectorX<T>* result) const {
  DRAKE_ASSERT(result->size() == z.size());

  Matrix3<T> Ap;  // the matrix A in energy_derivatives.md
  for (size_t p = 0; p < particles.num_particles(); ++p) {
    ComputeAp(p, z, particles, grid, dPdFs[p], &Ap);
    // sum over all nodes j, but only 27 of them will actually contribute
    const Vector3<int>& base_node_p = particles.GetBaseNodeAt(p);
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        for (int c = -1; c <= 1; ++c) {
          int node_local = 9 * (a + 1) + 3 * (b + 1) + (c + 1);  // 0-26
          size_t node_global =
              grid.To1DIndex(base_node_p + Vector3<int>(a, b, c));
          result->segment(3 * node_global, 3) +=
              dt * dt * particles.GetReferenceVolumeAt(p) * Ap *
              particles.GetElasticDeformationGradientAt(p).transpose() *
              particles.GetWeightGradientAt(p, node_local);
        }
      }
    }
  }
}

template <typename T>
void MpmTransfer<T>::ComputeAp(size_t p, const Eigen::VectorX<T>& z,
                               const Particles<T>& particles,
                               const SparseGrid<T>& grid,
                               const Eigen::Matrix<T, 9, 9>& dPdF,
                               Matrix3<T>* Ap) const {
  // compute B_p for each p
  Matrix3<T> B_p;
  B_p.setZero();
  // sum over all nodes j, but only 27 of them will actually contribute
  const Vector3<int>& base_node_p = particles.GetBaseNodeAt(p);
  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      for (int c = -1; c <= 1; ++c) {
        int node_local = 9 * (a + 1) + 3 * (b + 1) + (c + 1);  // 0-26
        size_t node_global =
            grid.To1DIndex(base_node_p + Vector3<int>(a, b, c));
        const Vector3<T>& z_j = z.segment(3 * node_global, 3);
        B_p += z_j * particles.GetWeightGradientAt(p, node_local).transpose() *
               particles.GetElasticDeformationGradientAt(p);
      }
    }
  }

  Ap->setZero();
  for (int alpha = 0; alpha < 3; ++alpha) {
    for (int beta = 0; beta < 3; ++beta) {
      for (int tau = 0; tau < 3; ++tau) {
        for (int sigma = 0; sigma < 3; ++sigma) {
          (*Ap)(alpha, beta) +=
              dPdF(alpha + 3 * beta, tau + 3 * sigma) * B_p(tau, sigma);
        }
      }
    }
  }
}

template class MpmTransfer<double>;
template class MpmTransfer<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
