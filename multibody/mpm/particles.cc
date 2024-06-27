#include "drake/multibody/mpm/particles.h"

namespace drake {
namespace multibody {
namespace mpm {

using drake::multibody::mpm::constitutive_model::ElastoPlasticModel;

template <typename T>
Particles<T>::Particles() {}

template <typename T>
void Particles<T>::AddParticle(
    const Vector3<T>& position, const Vector3<T>& velocity, const T& mass,
    const T& reference_volume, const Matrix3<T>& trial_deformation_gradient,
    const Matrix3<T>& elastic_deformation_gradient,
    std::unique_ptr<ElastoPlasticModel<T>> constitutive_model,
    const Matrix3<T>& B_matrix) {
  positions_.emplace_back(position);
  velocities_.emplace_back(velocity);
  initial_ids_.emplace_back(num_particles());
  masses_.emplace_back(mass);
  reference_volumes_.emplace_back(reference_volume);
  trial_deformation_gradients_.emplace_back(trial_deformation_gradient);
  elastic_deformation_gradients_.emplace_back(elastic_deformation_gradient);

  elastoplastic_models_.emplace_back(std::move(constitutive_model));

  Matrix3<T> P;
  elastoplastic_models_.back()->CalcFirstPiolaStress(
      Matrix3<T>::Identity(), elastic_deformation_gradient, &P);
  PK_stresses_.emplace_back(P);
  B_matrices_.emplace_back(B_matrix);

  temporary_scalar_field_.emplace_back();
  temporary_vector_field_.emplace_back();
  temporary_matrix_field_.emplace_back();
  temporary_base_nodes_.emplace_back();

  base_nodes_.emplace_back();
  weights_.emplace_back();

  permutation_.emplace_back();
  CheckAttributesSize();
}

template <typename T>
void Particles<T>::AddParticle(
    const Vector3<T>& position, const Vector3<T>& velocity,
    std::unique_ptr<ElastoPlasticModel<T>> constitutive_model, const T& mass,
    const T& reference_volume) {
  AddParticle(position, velocity, mass, reference_volume,
              Matrix3<T>::Identity(), Matrix3<T>::Identity(),
              std::move(constitutive_model), Matrix3<T>::Zero());
}

template <typename T>
void Particles<T>::Prepare(double h) {
  DRAKE_DEMAND(num_particles() > 0);
  // reserve space for batch_starts_ and batch_sizes
  // they can be upper-bounded by num_particles() as there is no empty batch
  batch_starts_.reserve(num_particles());
  batch_sizes_.reserve(num_particles());

  // 1) compute the base node for each particle
  for (size_t p = 0; p < num_particles(); ++p) {
    base_nodes_[p] = internal::ComputeBaseNodeFromPosition<T>(positions_[p], h);
  }
  // 2) sorts particle attributes
  // 2.1) get a sorted permutation
  std::iota(permutation_.begin(), permutation_.end(), 0);
  std::sort(permutation_.begin(), permutation_.end(),
            [this](size_t i1, size_t i2) {
              return internal::CompareIndex3DLexicographically(base_nodes_[i1],
                                                               base_nodes_[i2]);
            });
  // 2.2) shuffle particle data based on permutation
  Reorder(permutation_);  // including base_nodes_
  // 3) compute batch_starts_ and batch_sizes_
  batch_starts_.clear();
  batch_starts_.push_back(0);
  batch_sizes_.clear();
  Vector3<int> current_3d_index = base_nodes_[0];
  size_t count = 1;
  for (size_t p = 1; p < num_particles(); ++p) {
    if (base_nodes_[p] == current_3d_index) {
      ++count;
    } else {
      batch_starts_.push_back(p);
      batch_sizes_.push_back(count);
      // continue to next batch
      current_3d_index = base_nodes_[p];
      count = 1;
    }
  }
  batch_sizes_.push_back(count);

  DRAKE_DEMAND(batch_sizes_.size() == batch_starts_.size());

  // 4) compute d and dw
  for (size_t p = 0; p < num_particles(); ++p) {
    weights_[p].Reset(positions_[p], base_nodes_[p], h);
  }
  // 5) mark that the reordering has been done
  need_reordering_ = false;
}

template <typename T>
void Particles<T>::SplatToP2gPads(double h,
                                  std::vector<P2gPad<T>>* p2g_pads) const {
  DRAKE_DEMAND(!need_reordering_);
  p2g_pads->clear();  // remove existing data
  p2g_pads->resize(num_batches());
  for (size_t i = 0; i < num_batches(); ++i) {
    P2gPad<T>& p2g_pad = (*p2g_pads)[i];
    p2g_pad.SetZero();  // initialize to zero
    const size_t p_start = batch_starts_[i];
    const size_t p_end = p_start + batch_sizes_[i];
    for (size_t p = p_start; p < p_end; ++p) {
      weights_[p].SplatParticleDataToP2gPad(
          GetMassAt(p), GetPositionAt(p), GetVelocityAt(p),
          GetAffineMomentumMatrixAt(p, h), GetReferenceVolumeAt(p),
          GetPKStressAt(p), GetElasticDeformationGradientAt(p), base_nodes_[p],
          h, &p2g_pad);
    }
  }
}

template <typename T>
void Particles<T>::SplatStressToP2gPads(
    const std::vector<Matrix3<T>>& Ps, std::vector<P2gPad<T>>* p2g_pads) const {
  DRAKE_DEMAND(!need_reordering_);
  DRAKE_ASSERT(Ps.size() == num_particles());
  p2g_pads->resize(num_batches());
  for (size_t i = 0; i < num_batches(); ++i) {
    P2gPad<T>& p2g_pad = (*p2g_pads)[i];
    p2g_pad.SetZero();  // initialize to zero
    const size_t p_start = batch_starts_[i];
    const size_t p_end = p_start + batch_sizes_[i];
    for (size_t p = p_start; p < p_end; ++p) {
      weights_[p].SplatParticleStressToP2gPad(
          GetReferenceVolumeAt(p), Ps[p], GetElasticDeformationGradientAt(p),
          base_nodes_[p], &p2g_pad);
    }
  }
}

template <typename T>
void Particles<T>::UpdateBatchParticlesFromG2pPad(size_t batch_index, double dt,
                                                  const G2pPad<T>& g2p_pad) {
  DRAKE_ASSERT(batch_index < num_batches());
  const size_t p_start = batch_starts_[batch_index];
  const size_t p_end = p_start + batch_sizes_[batch_index];
  for (size_t p = p_start; p < p_end; ++p) {
    const ParticleVBGradV<T> particle_v_B_grad_v =
        weights_[p].AccumulateFromG2pPad(GetPositionAt(p), g2p_pad);
    SetVelocityAt(p, particle_v_B_grad_v.v);
    SetBMatrixAt(p, particle_v_B_grad_v.B_matrix);
    SetTrialDeformationGradientAt(
        p, (Matrix3<T>::Identity() + dt * particle_v_B_grad_v.grad_v) *
               GetElasticDeformationGradientAt(p));
  }
}

template <typename T>
void Particles<T>::WriteParticlesDataFromG2pPad(
    size_t batch_index, const G2pPad<T>& g2p_pad,
    ParticlesData<T>* particles_data) const {
  DRAKE_ASSERT(batch_index < num_batches());
  const size_t p_start = batch_starts_[batch_index];
  const size_t p_end = p_start + batch_sizes_[batch_index];
  for (size_t p = p_start; p < p_end; ++p) {
    const ParticleVBGradV<T> particle_v_B_grad_v =
        weights_[p].AccumulateFromG2pPad(GetPositionAt(p), g2p_pad);

    particles_data->particle_velocites_next[p] = particle_v_B_grad_v.v;
    particles_data->particle_B_matrices_next[p] = particle_v_B_grad_v.B_matrix;
    particles_data->particle_grad_v_next[p] = particle_v_B_grad_v.grad_v;
  }
}

template <typename T>
void Particles<T>::ComputeFsPsdPdFs(
    const std::vector<Matrix3<T>>& particle_grad_v, double dt, 
    std::vector<Matrix3<T>>* Fs, std::vector<Matrix3<T>>* Ps,
    std::vector<Eigen::Matrix<T, 9, 9>>* dPdFs, bool project_pd) const {
  DRAKE_ASSERT(particle_grad_v.size() == num_particles());
  for (size_t p = 0; p < num_particles(); ++p) {
    Matrix3<T>& F = (*Fs)[p];
    Matrix3<T>& P = (*Ps)[p];
    Eigen::Matrix<T, 9, 9>& dPdF = (*dPdFs)[p];

    F = (Matrix3<T>::Identity() + dt * particle_grad_v[p]) *
        GetElasticDeformationGradientAt(p);
    elastoplastic_models_[p]->CalcFirstPiolaStress(
        elastic_deformation_gradients_[p], F, &P);
    elastoplastic_models_[p]->CalcFirstPiolaStressDerivative(
        elastic_deformation_gradients_[p], F, &dPdF, project_pd);
  }
}

// TODO(zeshunzong): can be optimized a bit (computing only half)
template <typename T>
void Particles<T>::ComputePadHessianForOneBatch(
    size_t batch_i,
    const std::vector<Eigen::Matrix<T, 9, 9>>& dPdF_contractF0_contractF0,
    MatrixX<T>* pad_hessian) const {
  DRAKE_ASSERT(pad_hessian != nullptr);
  pad_hessian->resize(27 * 3, 27 * 3);
  pad_hessian->setZero();

  // loop over all particles in this batch_i
  for (size_t p = batch_starts_[batch_i];
       p < batch_starts_[batch_i] + batch_sizes_[batch_i]; ++p) {
    // loop over each i of 27 neighboring nodes in the numerator and each
    // dimension α
    for (size_t i = 0; i < 27; ++i) {
      for (size_t alpha = 0; alpha < 3; ++alpha) {
        // loop over each j of 27 neighboring nodes in the denominator and
        // each dimension rho
        for (size_t j = 0; j < 27; ++j) {
          for (size_t rho = 0; rho < 3; ++rho) {
            const Vector3<T>& gradNi_p = GetWeightGradientAt(p, i);
            const Vector3<T>& gradNj_p = GetWeightGradientAt(p, j);
            // The matrix for this particle, the indexing is A(α+3β, ρ+3γ)
            const Eigen::Matrix<T, 9, 9>& A_alphabeta_rhogamma =
                dPdF_contractF0_contractF0[p];
            // compute ∑ᵦᵧ A(α+3β, ρ+3γ) ⋅ ∇Nᵢ(xₚ)[β] ⋅ ∇Nⱼ(xₚ)[γ]
            for (size_t beta = 0; beta < 3; ++beta) {
              for (size_t gamma = 0; gamma < 3; ++gamma) {
                (*pad_hessian)(alpha + 3 * i, rho + 3 * j) +=
                    gradNi_p(beta) *
                    A_alphabeta_rhogamma(alpha + 3 * beta, rho + gamma * 3) *
                    gradNj_p(gamma) * GetReferenceVolumeAt(p);
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
std::vector<Eigen::Matrix<T, 9, 9>>
Particles<T>::ComputeDPDFContractF0ContractF0(
    const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs) const {
  std::vector<Eigen::Matrix<T, 9, 9>> dPdF_contractF0_contractF0(
      num_particles());
  for (size_t p = 0; p < num_particles(); ++p) {
    Eigen::Matrix<T, 9, 9>& matrix_p = dPdF_contractF0_contractF0[p];
    const Matrix3<T>& F0 = elastic_deformation_gradients_[p];
    // initialize
    matrix_p.setZero();
    // next contract with F0 twice
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int alpha = 0; alpha < 3; ++alpha) {
          for (int beta = 0; beta < 3; ++beta) {
            for (int gamma = 0; gamma < 3; ++gamma) {
              for (int rho = 0; rho < 3; ++rho) {
                matrix_p(alpha + 3 * beta, rho + 3 * gamma) +=
                    dPdFs[p](alpha + 3 * i, rho + 3 * j) * F0(gamma, j) *
                    F0(beta, i);
              }
            }
          }
        }
      }
    }
  }
  return dPdF_contractF0_contractF0;
}

template <typename T>
void Particles<T>::Reorder(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles());
  for (size_t i = 0; i < num_particles() - 1; ++i) {
    // don't need to sort the last element if the first (n-1) elements have
    // already been sorted
    size_t ind = new_order[i];
    // the i-th element should be placed at ind-th position
    if (ind < i) {
      // its correct position is before i. In this case, the element must have
      // already been swapped to a position after i. find out where it has been
      // swapped to.
      while (ind < i) {
        ind = new_order[ind];
      }
    }
    // after this operation, ind is either equal to i or larger than i
    if (ind == i) {
      // at its correct position, nothing needs to be done
      continue;
    } else if (ind > i) {
      // TODO(zeshunzong): update this as more attributes are added
      std::swap(initial_ids_[i], initial_ids_[ind]);
      std::swap(masses_[i], masses_[ind]);
      std::swap(reference_volumes_[i], reference_volumes_[ind]);

      std::swap(positions_[i], positions_[ind]);
      std::swap(velocities_[i], velocities_[ind]);

      std::swap(elastic_deformation_gradients_[i],
                elastic_deformation_gradients_[ind]);
      std::swap(trial_deformation_gradients_[i],
                trial_deformation_gradients_[ind]);
      std::swap(B_matrices_[i], B_matrices_[ind]);

      std::swap(base_nodes_[i], base_nodes_[ind]);

      std::swap(elastoplastic_models_[i], elastoplastic_models_[ind]);
    } else {
      DRAKE_UNREACHABLE();
    }
  }
}

// TODO(zeshunzong):swap elastoplastic_models_
template <typename T>
void Particles<T>::Reorder2(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles());

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_int_field_[i] = initial_ids_[new_order[i]];
  }
  initial_ids_.swap(temporary_int_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_scalar_field_[i] = masses_[new_order[i]];
  }
  masses_.swap(temporary_scalar_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_scalar_field_[i] = reference_volumes_[new_order[i]];
  }
  reference_volumes_.swap(temporary_scalar_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_vector_field_[i] = positions_[new_order[i]];
  }
  positions_.swap(temporary_vector_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_vector_field_[i] = velocities_[new_order[i]];
  }
  velocities_.swap(temporary_vector_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_matrix_field_[i] = trial_deformation_gradients_[new_order[i]];
  }
  trial_deformation_gradients_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_matrix_field_[i] = elastic_deformation_gradients_[new_order[i]];
  }
  elastic_deformation_gradients_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_matrix_field_[i] = B_matrices_[new_order[i]];
  }
  B_matrices_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_base_nodes_[i] = base_nodes_[new_order[i]];
  }
  base_nodes_.swap(temporary_base_nodes_);
}

template <typename T>
void Particles<T>::CheckAttributesSize() const {
  // by construction num_particles() = positions_.size()
  DRAKE_DEMAND(num_particles() == velocities_.size());
  DRAKE_DEMAND(num_particles() == initial_ids_.size());
  DRAKE_DEMAND(num_particles() == masses_.size());
  DRAKE_DEMAND(num_particles() == reference_volumes_.size());
  DRAKE_DEMAND(num_particles() == trial_deformation_gradients_.size());
  DRAKE_DEMAND(num_particles() == elastic_deformation_gradients_.size());
  DRAKE_DEMAND(num_particles() == B_matrices_.size());

  DRAKE_DEMAND(num_particles() == temporary_scalar_field_.size());
  DRAKE_DEMAND(num_particles() == temporary_vector_field_.size());
  DRAKE_DEMAND(num_particles() == temporary_matrix_field_.size());

  DRAKE_DEMAND(num_particles() == base_nodes_.size());
  DRAKE_DEMAND(num_particles() == weights_.size());

  DRAKE_DEMAND(num_particles() == permutation_.size());
}

template <typename T>
internal::MassAndMomentum<T> Particles<T>::ComputeTotalMassMomentum() const {
  internal::MassAndMomentum<T> total_mass_momentum{};
  for (size_t p = 0; p < num_particles(); ++p) {
    total_mass_momentum.total_mass += masses_[p];
    total_mass_momentum.total_momentum += masses_[p] * velocities_[p];
    total_mass_momentum.total_angular_momentum +=
        masses_[p] *
        (positions_[p].cross(velocities_[p]) +
         internal::ContractionWithLeviCivita<T>(B_matrices_[p].transpose()));
  }

  return total_mass_momentum;
}

template class Particles<double>;
template class Particles<AutoDiffXd>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
