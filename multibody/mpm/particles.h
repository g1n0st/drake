#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/interpolation_weights.h"

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
   * <!-- TODO(zeshunzong): More attributes will come later. -->
   * <!-- TODO(zeshunzong): Do we always start from rest shape? so F=I? -->
   */
  void AddParticle(const Vector3<T>& position, const Vector3<T>& velocity,
                   const T& mass, const T& reference_volume,
                   const Matrix3<T>& trial_deformation_gradient,
                   const Matrix3<T>& elastic_deformation_gradient,
                   const Matrix3<T>& B_matrix);

  /**
   * Adds (appends) a particle into Particles with given properties. We assume
   * that the particles start from rest shape, so deformation gradient is
   * identity and B_matrix is zero.
   */
  void AddParticle(const Vector3<T>& position, const Vector3<T>& velocity,
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

  void SplatToPads(double h, std::vector<Pad<T>>* pads) const {
    DRAKE_DEMAND(!need_reordering_);
    pads->resize(num_batches());
    for (size_t i = 0; i < num_batches(); ++i) {
      Pad<T>& pad = (*pads)[i];
      const size_t p_start = batch_starts_[i];
      const size_t p_end = p_start + batch_sizes_[i];
      for (size_t p = p_start; p < p_end; ++p) {
        weights_[p].SplatParticleDataToPad(
            GetMassAt(p), GetPositionAt(p), GetVelocityAt(p),
            GetAffineMomentumMatrixAt(p, h), GetReferenceVolumeAt(p),
            GetPKStressAt(p), GetElasticDeformationGradientAt(p),
            base_nodes_[i], h, &pad);
      }
    }
  }

  /**
   * Advects each particle's position x_p by dt*v_p, where v_p is particle's
   * velocity.
   */
  void AdvectParticles(double dt) {
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
  const std::vector<T>& masses() const { return masses_; }
  const std::vector<T>& reference_volumes() const { return reference_volumes_; }
  const std::vector<Matrix3<T>>& trial_deformation_gradients() const {
    return trial_deformation_gradients_;
  }
  const std::vector<Matrix3<T>>& elastic_deformation_gradients() const {
    return elastic_deformation_gradients_;
  }
  const std::vector<Matrix3<T>>& B_matrices() const { return B_matrices_; }

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
   * TODO(zeshunzong) implement it. may also change return type
   */
  Matrix3<T> GetPKStressAt(size_t p) const {
    DRAKE_ASSERT(p < num_particles());
    return Matrix3<T>::Zero();
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
  void SetTrialDeformationGradient(size_t p, const Matrix3<T>& F_trial_in) {
    DRAKE_ASSERT(p < num_particles());
    trial_deformation_gradients_[p] = F_trial_in;
  }

  /**
   * Sets the elastic deformation gradient at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetElasticDeformationGradient(size_t p, const Matrix3<T>& FE_in) {
    DRAKE_ASSERT(p < num_particles());
    elastic_deformation_gradients_[p] = FE_in;
  }

  /**
   * Sets the B_matrix at p-th particle from input.
   * @pre 0 <= p < num_particles()
   */
  void SetBMatrix(size_t p, const Matrix3<T>& B_matrix) {
    DRAKE_ASSERT(p < num_particles());
    B_matrices_[p] = B_matrix;
  }
  /**
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
   */
  const std::vector<Vector3<int>>& base_nodes() { return base_nodes_; }
  const std::vector<size_t>& batch_starts() { return batch_starts_; }
  const std::vector<size_t>& batch_sizes() { return batch_sizes_; }

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
  // TODO(zeshunzong): May need to reorder more attributes as more
  // attributes are added.
  void Reorder(const std::vector<size_t>& new_order);

  // Performs the same function as Reorder but in a constant O(n) way.
  // TODO(zeshunzong): Technically we can reduce the number of copies
  // introducing a flag and alternatingly return the (currently) ordered
  // attribute. Since we haven't decided which algorithm to use, for clarity
  // let's defer this for future.
  // TODO(zeshunzong): May need to reorder more attributes as more
  // attributes are added.
  void Reorder2(const std::vector<size_t>& new_order);

  // particle-wise data
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

  // TODO(zeshunzong): Consider make struct Scratch and put the buffer data
  // inside scratch for better clarity. for reorder only
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