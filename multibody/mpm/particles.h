#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"
#include "drake/multibody/mpm/internal/mass_and_momentum.h"
#include "drake/multibody/mpm/interpolation_weights.h"
#include "drake/multibody/mpm/particles_data.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A Particles class holding particle states as several std::vectors.
 * Each particle carries its own position, velocity, mass, volume, etc.
 *
 * The Material Point Method (MPM) consists of a set of particles (implemented
 * in this class) and a background Eulerian grid (implemented in sparse_grid.h).
 * At each time step, particle masses and momentums are transferred to the grid
 * nodes via a B-spline interpolation function (implemented in
 * internal::b_spline.h).
 *
 * For computation purpose, particles clustered around one grid node are
 * classified into one batch (the batch is marked by their center grid node).
 * Each particle belongs to *exactly* one batch.
 * After executing Prepare(), the batches and particles look like the
 * following (schematically in 2D).
 *
 *           . ---- . ---- ~ ---- .
 *           |      |      |9     |
 *           |2     |64    |      |
 *           x ---- o ---- + ---- #
 *           |     3| 5    |7    8|
 *           |    01|      |      |
 *           . ---- * ---- . ---- .
 *
 * @note particles are sorted lexicographically based on their base nodes.
 * Therefore, within a batch where the particles share a common base node,
 * there is no fixed ordering for the particles (but the ordering is
 * deterministic). base_nodes_[0] = base_nodes_[1] = (the 3d index of) *
 * base_nodes_[2] = x
 * base_nodes_[3] = base_nodes_[4] = base_nodes_[5] = base_nodes_[6] = o
 * base_nodes_[7] = +
 * base_nodes_[8] = #
 * base_nodes_[9] = ~
 * There are a total of num_batches() = six batches.
 * batch_sizes_[0] = number of particles around * = 2
 * batch_sizes_[1] = number of particles around x = 1
 * batch_sizes_[2] = number of particles around o = 4
 * batch_sizes_[3] = number of particles around + = 1
 * batch_sizes_[4] = number of particles around # = 1
 * batch_sizes_[5] = number of particles around ~ = 1
 * @note the sum of all batch_sizes is equal to num_particles()
 *
 * batch_starts_[0] = the first particle in batch * = 0
 * batch_starts_[1] = the first particle in batch x = 2
 * batch_starts_[2] = the first particle in batch o = 4
 * batch_starts_[3] = the first particle in batch + = 7
 * batch_starts_[4] = the first particle in batch # = 8
 * batch_starts_[5] = the first particle in batch ~ = 9
 *
 * num_batches() = 6
 */
template <typename T>
class Particles {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Particles);

  /**
   * Creates a Particles container with 0 particle. All std::vector<> are set
   * to length zero. This, working together with AddParticle(), should be the
   * default version to insert particles one after another.
   */
  Particles();

  /**
   *  Adds (appends) a particle into Particles with given properties.
   */
  void AddParticle(
      const Vector3<T>& position, const Vector3<T>& velocity, const T& mass,
      const T& reference_volume, const Matrix3<T>& trial_deformation_gradient,
      const Matrix3<T>& elastic_deformation_gradient,
      std::unique_ptr<mpm::constitutive_model::ElastoPlasticModel<T>>
          constitutive_model,
      const Matrix3<T>& B_matrix);

  /**
   * Adds (appends) a particle into Particles with given properties. We assume
   * that the particles start from rest shape, so deformation gradient is
   * identity, stress is zero, and B_matrix is zero.
   */
  void AddParticle(
      const Vector3<T>& position, const Vector3<T>& velocity,
      std::unique_ptr<mpm::constitutive_model::ElastoPlasticModel<T>>
          constitutive_model,
      const T& mass, const T& reference_volume);

  /**
   * To perform ParticlesToGrid transfer and GridToParticles transfer (as
   * implemented in mpm_transfer.h), one needs the up-to-date interpolation
   * weights. Further, to speedup data accessing, particles are also sorted
   * lexicographically with respect to their positions. This function calculates
   * the interpolation weights and reorders particles based on current particle
   * positions, thus providing all necessary ingredients (including
   * batch_starts_ and batch_sizes_) for transfers.
   * @note both the particle ordering and the interpolation weights only depend
   * on particle positions. Hence, this function should be called whenever the
   * particle positions change.
   * @note a flag need_reordering_ is (temporarily) used to keep track of the
   * dependency on particle positions. It will be set to false when this
   * function executes.
   *
   * To be more specific, the following operations are performed:
   * 1) Computes the base node for each particle.
   * 2) Sorts particle attributes lexicographically by their base nodes
   * positions.
   * 3) Computes batch_starts_ and batch_sizes_.
   * 4) Computes weights and weight gradients
   * 5) Marks that the reordering has been done.
   */
  void Prepare(double h);

  void ResetToInitialOrder() {
    std::iota(permutation_.begin(), permutation_.end(), 0);
    std::sort(permutation_.begin(), permutation_.end(),
              [this](size_t i1, size_t i2) {
                return initial_ids_[i1] < initial_ids_[i2];
              });
    Reorder(permutation_);
  }

  /**
   * Splats particle data to p2g_pads. Particles in the same batch will have
   * their data splatted to the same p2g_pad.
   * @note p2g_pads will be cleared and resized to num_batches().
   * @pre !NeedReordering()
   * @pre h > 0
   */
  void SplatToP2gPads(double h, std::vector<P2gPad<T>>* p2g_pads) const;

  /**
   * Splats the first Piola-Kirchhoff stress for each particle from the input
   * `Ps` to `p2g_pads`. Particles in the same batch will have their stresses
   * splatted to the same pad.
   * @note p2g_pads will be cleared and resized to num_batches().
   * @pre !NeedReordering()
   * @pre PK_stress_all.size() == num_particles().
   */
  void SplatStressToP2gPads(const std::vector<Matrix3<T>>& Ps,
                            std::vector<P2gPad<T>>* p2g_pads) const;

  /**
   * Advects each particle's position x_p by dt*v_p, where v_p is particle's
   * velocity.
   * @pre particles must not need reordering. That is, this function cannot be
   * called multiple times consecutively.
   */
  void AdvectParticles(double dt) {
    DRAKE_DEMAND(!need_reordering_);
    for (size_t p = 0; p < num_particles(); ++p) {
      positions_[p] += dt * velocities_[p];
    }
    need_reordering_ = true;
  }

  size_t num_particles() const { return positions_.size(); }
  size_t num_batches() const { return batch_starts_.size(); }

  // Disambiguation:
  // positions: a std::vector holding position of all particles.
  // position:  the position of a particular particle. This shall always be
  // associated with a particle index p.
  // This naming rule applies to all class attributes.

  const std::vector<Vector3<T>>& positions() const { return positions_; }
  const std::vector<Vector3<T>>& velocities() const { return velocities_; }
  const std::vector<int>& initial_ids() const { return initial_ids_; }
  const std::vector<T>& masses() const { return masses_; }
  const std::vector<T>& reference_volumes() const { return reference_volumes_; }
  const std::vector<Matrix3<T>>& trial_deformation_gradients() const {
    return trial_deformation_gradients_;
  }
  const std::vector<Matrix3<T>>& elastic_deformation_gradients() const {
    return elastic_deformation_gradients_;
  }
  const std::vector<Matrix3<T>>& B_matrices() const { return B_matrices_; }

  void SetBMatrices(const std::vector<Matrix3<T>>& B_matrices_in) {
    DRAKE_ASSERT(B_matrices_in.size() == num_particles());
    B_matrices_ = B_matrices_in;
  }

  void SetVelocities(const std::vector<Vector3<T>>& velocities_in) {
    DRAKE_ASSERT(velocities_in.size() == num_particles());
    velocities_ = velocities_in;
  }

  /**
   * Returns the position of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Vector3<T>& GetPositionAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return positions_[p];
  }

  /**
   * Returns the velocity of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Vector3<T>& GetVelocityAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return velocities_[p];
  }

  /**
   * Returns the mass of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const T& GetMassAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return masses_[p];
  }

  /**
   * Returns the reference volume of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const T& GetReferenceVolumeAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return reference_volumes_[p];
  }

  /**
   * Returns the trial deformation gradient of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Matrix3<T>& GetTrialDeformationGradientAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return trial_deformation_gradients_[p];
  }

  /**
   * Returns the elastic deformation gradient of p-th particle.
   * @pre 0 <= p < num_particles()
   */
  const Matrix3<T>& GetElasticDeformationGradientAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return elastic_deformation_gradients_[p];
  }

  /**
   * Returns the first Piola Kirchhoff stress computed for the p-th particle.
   */
  const Matrix3<T>& GetPKStressAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return PK_stresses_[p];
  }

  /**
   * Returns the B_matrix of p-th particle. B_matrix is part of the affine
   * momentum matrix C as
   * v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p).
   * See (173) in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   * @pre 0 <= p < num_particles()
   */
  const Matrix3<T>& GetBMatrixAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return B_matrices_[p];
  }

  /**
   * Returns the affine momentum matrix C of the p-th particle.
   * C_p = B_p * D_p^-1. In quadratic B-spline, D_p is a diagonal matrix all
   * diagonal elements being 0.25*h*h.
   */
  Matrix3<T> GetAffineMomentumMatrixAt(size_t p, double h) const {
    DRAKE_ASSERT(p < num_particles());
    Matrix3<T> C_matrix = B_matrices_[p] * (4.0 / h / h);
    return C_matrix;
  }

  /**
   * Sets the velocity at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetVelocityAt(size_t p, const Vector3<T>& velocity) {
    DRAKE_ASSERT(p < num_particles());
    velocities_[p] = velocity;
  }

  /**
   * Sets the trial deformation gradient at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetTrialDeformationGradientAt(size_t p, const Matrix3<T>& F_trial_in) {
    DRAKE_ASSERT(p < num_particles());
    trial_deformation_gradients_[p] = F_trial_in;
  }

  /**
   * Sets the elastic deformation gradient at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetElasticDeformationGradientAt(size_t p, const Matrix3<T>& FE_in) {
    DRAKE_ASSERT(p < num_particles());
    elastic_deformation_gradients_[p] = FE_in;
  }

  /**
   * Sets the B_matrix at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetBMatrixAt(size_t p, const Matrix3<T>& B_matrix) {
    DRAKE_ASSERT(p < num_particles());
    B_matrices_[p] = B_matrix;
  }

  /**
   * For particles in the same batch denoted by batch_index, updates their
   * velocities, B-matrices, and *trial* deformation gradient matrices, using
   * the revalent grid data stored in g2p_pad. Particles in the same batch
   * (i.e. with the same base_node) are updated from the same 27 grid nodes (a
   * 3by3by3 cube centered at that base_node). The information for the 27 grid
   * nodes are stored in g2p_pad.
   * @pre batch_index < num_batches().
   * @note particle positions are NOT updated in this function!
   */
  void UpdateBatchParticlesFromG2pPad(size_t batch_index, double dt,
                                      const G2pPad<T>& g2p_pad);

  /**
   * Writes pad data from g2p_pad to scratch for time integration.
   * @pre data attributes in particles_data all have size num_particles()
   */
  void WriteParticlesDataFromG2pPad(size_t batch_index,
                                    const G2pPad<T>& g2p_pad,
                                    ParticlesData<T>* particles_data) const;

  /**
   * Updates trial deformation gradient from grad_v, the formula is
   * Fₚⁿ⁺¹ = (I + Δ t ⋅ ∑ᵢ vᵢⁿ⁺¹ (∇ wⁿᵢₚ)ᵀ) Fⁿₚ.
   * See eqn (181) in
   * https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf.
   */
  void UpdateTrialDeformationGradients(
      double dt, const std::vector<Matrix3<T>>& particle_grad_v) {
    for (size_t p = 0; p < num_particles(); ++p) {
      SetTrialDeformationGradientAt(
          p, (Matrix3<T>::Identity() + dt * particle_grad_v[p]) *
                 GetElasticDeformationGradientAt(p));
    }
  }

  /**
   * Updates elastic_F = ReturnMap(F_trial)
   *         PK_stress = ComputeStress(elastic_F)
   */
  void UpdateElasticDeformationGradientsAndStresses() {
    Matrix3<T> FE;
    for (size_t p = 0; p < num_particles(); ++p) {
      elastoplastic_models_[p]->CalcFEFromFtrial(
          elastic_deformation_gradients_[p], trial_deformation_gradients_[p],
          &FE);
      elastoplastic_models_[p]->CalcFirstPiolaStress(
          elastic_deformation_gradients_[p], FE, &(PK_stresses_[p]));
      elastic_deformation_gradients_[p] = FE;
    }
  }

  const std::vector<Vector3<int>>& base_nodes() const { return base_nodes_; }
  const std::vector<size_t>& batch_starts() const { return batch_starts_; }
  const std::vector<size_t>& batch_sizes() const { return batch_sizes_; }
  const Vector3<int>& GetBaseNodeAt(size_t p) const { return base_nodes_[p]; }

  bool NeedReordering() const { return need_reordering_; }

  /**
   * Computes the mass and momentum of the continuum by summing over all
   * particles.
   * For angular momentum computation, see Jiang, C., Schroeder, C., & Teran, J.
   * (2017). An angular momentum conserving affine-particle-in-cell method.
   * Journal of Computational Physics, 338, 137-164. Section 5.3.
   * https://math.ucdavis.edu/~jteran/papers/JST17.pdf
   */
  internal::MassAndMomentum<T> ComputeTotalMassMomentum() const;

  /**
   * Computes the total elastic energy of all particles, using the elastic
   * deformation gradients of all particles. See accompanied
   * energy_derivatives.md for formula.
   * @note F_all stores the *elastic* deformation gradient for all particles
   * @pre F_all.size() == num_particles()   */
  T ComputeTotalElasticEnergy(const std::vector<Matrix3<T>>& F_all) const {
    T sum = 0;
    for (size_t p = 0; p < num_particles(); ++p) {
      sum += reference_volumes_[p] *
             elastoplastic_models_[p]->CalcStrainEnergyDensity(
                 elastic_deformation_gradients_[p], F_all[p]);
    }
    return sum;
  }

  /**
   * Computes the deformation info for computing elastic energy and
   * its derivatives. More specifically, for each particle p, given grad_v of
   * this particle, we compute:
   * F = (I + dt * grad_v) * F₀.
   * P = CalcFirstPiolaStress(F).
   * dPdF = CalcFirstPiolaStressDerivative(F).
   */
  void ComputeFsPsdPdFs(const std::vector<Matrix3<T>>& particle_grad_v,
                        double dt, std::vector<Matrix3<T>>* Fs,
                        std::vector<Matrix3<T>>* Ps,
                        std::vector<Eigen::Matrix<T, 9, 9>>* dPdFs,
                        bool project_pd = false) const;

  /**
   * Computes the part of elastic hessian (denoted `pad_hessian`) contributed by
   * particles in the batch_i-th batch. This will be non-zero for the 27 grid
   * nodes neighboring to the common base_node that this batch of particles
   * share. Thus, the result is stored in a (27*3)by(27*3) matrix.
   * pad_hessian(3*i+α, 3*j+β) = d²e / dxᵢₐdxⱼᵦ, where e is the elastic energy
   * restricted to the particles in this batch, and i and j are the local index
   * relative to the shared base_node (0-26) of the grid nodes.
   */
  void ComputePadHessianForOneBatch(
      size_t batch_i,
      const std::vector<Eigen::Matrix<T, 9, 9>>& dPdF_contractF0_contractF0,
      MatrixX<T>* pad_hessian) const;

  /**
   * For each particle, computes dPdF:F₀:F₀, and return the stored result.
   * The formula is resultₐᵦᵨᵧ = ∑ⱼₗ dPdFₐᵢᵨⱼ * F₀ᵧⱼ * F₀ᵦᵢ.
   * @note recall for a 4D tensor stored as a 9by9 matrix, the ordering is Aᵢⱼₖₗ
   * = A(i+3j, k+3l)
   */
  std::vector<Eigen::Matrix<T, 9, 9>> ComputeDPDFContractF0ContractF0(
      const std::vector<Eigen::Matrix<T, 9, 9>>& dPdFs) const;

  /**
   * Returns the weight gradient ∇Nᵢ(xₚ), where i is the local index of a
   * neighbor grid node.
   * @pre p < num_particles().
   * @pre neighbor_grid < 27.
   */
  const Vector3<T>& GetWeightGradientAt(size_t p, size_t neighbor_node) const {
    DRAKE_ASSERT(p < num_particles());
    return weights_[p].GetWeightGradientAt(neighbor_node);
  }

  /**
   * Returns the weight Nᵢ(xₚ), where i is the local index of a neighbor grid
   * node.
   * @pre p < num_particles().
   * @pre neighbor_grid < 27.
   */
  const T& GetWeightAt(size_t p, size_t neighbor_node) const {
    DRAKE_ASSERT(p < num_particles());
    return weights_[p].GetWeightAt(neighbor_node);
  }

  void ScaleYoungsModulus(size_t p, T scale) {
    elastoplastic_models_[p]->set_E(elastoplastic_models_[p]->youngs_modulus() *
                                    scale);
  }

 private:
  // Ensures that all attributes (std::vectors) have correct size. This only
  // needs to be called when new particles are added.
  // TODO(zeshunzong): more attributes may come later.
  // TODO(zeshunzong): technically this can be removed. I decided to keep it
  // only during the current stage where we don't have a final say of the number
  // of attributes we want.
  void CheckAttributesSize() const;

  // Permutes all states in Particles with respect to the index set new_order.
  // e.g. given new_order = [2; 0; 1], and the original particle states are
  // denoted by [p0; p1; p2]. The new particles states after calling Reorder()
  // will be [p2; p0; p1].
  // @pre new_order is a permutation of [0, ..., num_particles-1]
  // @note this algorithm uses index-chasing and might be O(n^2) in worst case.
  // TODO(zeshunzong): this algorithm is insanely fast for "simple"
  // permutations. A standard O(n) algorithm is implemented below in Reorder2().
  // We should decide on which one to choose once the whole MPM pipeline is
  // finished.
  void Reorder(const std::vector<size_t>& new_order);

  // Performs the same function as Reorder but in a constant O(n) way.
  // TODO(zeshunzong): Technically we can reduce the number of copies
  // introducing a flag and alternatingly return the (currently) ordered
  // attribute. Since we haven't decided which algorithm to use, for clarity
  // let's defer this for future.
  // TODO(zeshunzong):swap elastoplastic_models_
  void Reorder2(const std::vector<size_t>& new_order);

  // For each particle p, computes dPdF_contractF0_contractF0[p](3β+α, 3γ+ρ) =
  // [∑ᵢⱼ (dPₐᵢ/dFᵨⱼ) * Fᵧⱼ * Fᵦᵢ] * Vₚ⁰ i and j are θ and ϕ in accompanied
  // ElasticEnergyDerivatives.md. The result is stored as A_(alpha beta, rho
  // gamma), where alpha and rho are the outer (block) row and column indices.

  // particle-wise data
  std::vector<int> initial_ids_{};
  std::vector<Vector3<T>> positions_{};
  std::vector<Vector3<T>> velocities_{};
  std::vector<T> masses_{};
  std::vector<T> reference_volumes_{};

  std::vector<Matrix3<T>> trial_deformation_gradients_{};
  std::vector<Matrix3<T>> elastic_deformation_gradients_{};

  // The affine matrix B_p in APIC
  // B_matrix is part of the affine momentum matrix C as
  // v_i = v_p + C_p (x_i - x_p) = v_p + B_p D_p^-1 (x_i - x_p).
  std::vector<Matrix3<T>> B_matrices_{};

  // First Piola Kirchhoff stress. Do not reoder this as this is computed from a
  // reordered elastoplastic model each time!
  std::vector<Matrix3<T>> PK_stresses_{};

  std::vector<
      copyable_unique_ptr<mpm::constitutive_model::ElastoPlasticModel<T>>>
      elastoplastic_models_{};

  // TODO(zeshunzong): Consider make struct Scratch and put the buffer data
  // inside scratch for better clarity. for reorder only
  std::vector<int> temporary_int_field_{};
  std::vector<T> temporary_scalar_field_{};
  std::vector<Vector3<T>> temporary_vector_field_{};
  std::vector<Matrix3<T>> temporary_matrix_field_{};
  std::vector<Vector3<int>> temporary_base_nodes_{};

  // particle-wise batch info
  // base_nodes_[i] is the 3d index of the base node of the i-th particle.
  // size = num_particles()
  std::vector<Vector3<int>> base_nodes_{};

  // size = num_particles()
  // but this does not need to be sorted, as whenever sorting is required, this
  // means particle positions change, so weights need to be re-computed
  std::vector<InterpolationWeights<T>> weights_{};

  // batch_starts_[i] is the index of the first particle in the i-th batch.
  // size = total number of batches, <= num_particles().
  std::vector<size_t> batch_starts_;
  // batch_sizes_[i] is the number of particles in the i-th batch.
  // size = total number of batches, <= num_particles().
  std::vector<size_t> batch_sizes_;

  // a flag to track necessary updates when particle positions change
  // particle positions can only be changed in
  // 1) AddParticle()
  // 2) AdvectParticles()
  bool need_reordering_ = true;

  // intermediary variable used for sorting particles
  std::vector<size_t> permutation_;
};
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
