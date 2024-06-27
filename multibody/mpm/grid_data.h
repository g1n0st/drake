#pragma once

#include <unordered_set>
#include <vector>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

/**
 * A container for the kinematic data that will be stored on grid nodes.
 * In particular, they are
 *      mass on each active node
 *      momentum on each active node
 *      velocity on each active node
 *      force on each active node
 */
template <typename T>
class GridData {
 public:
  void Reserve(size_t capacity) {
    masses_.reserve(capacity);
    momentums_.reserve(capacity);
    velocities_.reserve(capacity);
    forces_.reserve(capacity);
  }

  size_t num_active_nodes() const { return masses_.size(); }

  /**
   * Resets the containers to the correct size.
   * Sets all values to zero.
   */
  void Reset(size_t num_active_nodes) {
    masses_.resize(num_active_nodes);
    momentums_.resize(num_active_nodes);
    velocities_.resize(num_active_nodes);
    forces_.resize(num_active_nodes);
    std::fill(masses_.begin(), masses_.end(), 0.0);
    std::fill(momentums_.begin(), momentums_.end(), Vector3<T>::Zero());
    std::fill(forces_.begin(), forces_.end(), Vector3<T>::Zero());
    std::fill(velocities_.begin(), velocities_.end(), Vector3<T>::Zero());
  }

  /**
   * Increments velocity to each grid node component.
   */
  void AddDG(const Eigen::VectorX<T>& dG) {
    DRAKE_ASSERT(static_cast<int>(dG.size()) ==
                 static_cast<int>(num_active_nodes() * 3));
    for (size_t i = 0; i < num_active_nodes(); ++i) {
      velocities_[i] += dG.segment(3 * i, 3);
    }
  }

  /**
   * Adds mass, momentum, and force to the node at index_1d.
   */
  void AccumulateAt(size_t index_1d, const T& mass, const Vector3<T>& momentum,
                    const Vector3<T>& force) {
    DRAKE_ASSERT(index_1d < masses_.size());
    masses_[index_1d] += mass;
    momentums_[index_1d] += momentum;
    forces_[index_1d] += force;
  }

  /**
   * Computes the velocity for momentum for each active grid node.
   * velocity = momentum / mass.
   * @note usually the mass will be non-zero, except when all particles fall
   * right on the boundary of the support of B-spline kernel for this node.
   * @note when mass is zero, momentum will also be zero, and velocity is
   * clearly zero. We add an if statement to prevent 0/0.
   */
  void ComputeVelocitiesFromMomentums() {
    for (size_t i = 0; i < masses_.size(); ++i) {
      if (masses_[i] == 0.0) {
        velocities_[i].setZero();
      } else {
        velocities_[i] = momentums_[i] / masses_[i];
      }
    }
  }

  void ApplyExplicitForceImpulsesToVelocities(const T& dt) {
    for (size_t i = 0; i < masses_.size(); ++i) {
      velocities_[i] += forces_[i] / masses_[i] * dt;
    }
  }

  /**
   * @pre index_1d < num_active_nodes()
   */
  const Vector3<T>& GetVelocityAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < velocities_.size());
    return velocities_[index_1d];
  }

  void GetFlattenedVelocities(VectorX<T>* result) const {
    result->resize(3 * num_active_nodes());
    for (size_t i = 0; i < num_active_nodes(); ++i) {
      result->segment(3 * i, 3) = GetVelocityAt(i);
    }
  }

  /**
   * @pre index_1d < num_active_nodes()
   */
  const T& GetMassAt(size_t index_1d) const {
    DRAKE_ASSERT(index_1d < masses_.size());
    return masses_[index_1d];
  }

  const std::vector<T>& masses() const { return masses_; }
  const std::vector<Vector3<T>>& momentums() const { return momentums_; }
  const std::vector<Vector3<T>>& velocities() const { return velocities_; }

  void SetVelocities(const std::vector<Vector3<T>>& velocities) {
    velocities_ = velocities;
  }

  void SetVelocities(const VectorX<T>& velocities) {
    for (size_t i = 0; i < num_active_nodes(); ++i) {
      velocities_[i] = velocities.segment(3 * i, 3);
    }
  }

  void SetVelocityAt(const Vector3<T>& v, size_t node) {
    velocities_[node] = v;
  }

  void ExtractVelocitiesFromIndices(const std::unordered_set<int>& indices,
                                    VectorX<T>* result) const {
    result->resize(indices.size() * 3);
    int count = 0;
    for (int i = 0; i < static_cast<int>(num_active_nodes()); ++i) {
      if (indices.count(i)) {
        result->segment(3 * count, 3) = GetVelocityAt(i);
        ++count;
      }
    }
  }

  void AddDKineticEnergyDV(std::vector<Vector3<T>>* result) const {
    DRAKE_ASSERT(result != nullptr);
    DRAKE_ASSERT(result->size() == num_active_nodes());

    for (size_t i = 0; i < num_active_nodes(); ++i) {
      (*result)[i] += GetMassAt(i) * velocities_[i];
    }
  }

  void AddDGravitationalEnergyDV(std::vector<Vector3<T>>* result) const {
    DRAKE_ASSERT(result != nullptr);
    DRAKE_ASSERT(result->size() == num_active_nodes());
    for (size_t i = 0; i < num_active_nodes(); ++i) {
      (*result)[i] += GetMassAt(i) * velocities_[i];
    }
  }

  void ProjectionGround(const std::vector<size_t>& collision_nodes,
                        bool sticky) {
    for (auto node_idx : collision_nodes) {
      if (sticky) {
        velocities_[node_idx].setZero();
      } else {
        velocities_[node_idx](2) = 0.0;
      }
    }
  }

 private:
  std::vector<T> masses_;
  std::vector<Vector3<T>> momentums_;
  std::vector<Vector3<T>> velocities_;
  std::vector<Vector3<T>> forces_;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
