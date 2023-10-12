#pragma once

#include <array>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// Generates a set of samples within a bounding box (including boundary) such
// that no two samples are closer than a user-specified distance r. The bounding
// box is specified as x_min[0], x_max[0]] X [x_min[1], x_max[1]] X [x_min[2],
// x_max[2]]. See
// https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf and
// https://github.com/thinks/poisson-disk-sampling/tree/master for details.
// @pre r > 0.
// @pre x_max[i] > x_min[i] for i = 0, 1, 2.
// @note The algorithm is determinstic, as the random seed is fixed. It will
// generate the same distribution for the same input.
// @note The algorithm stops when we can't sample another point that's more than
// r away from all existing points.
// @tparam T double or float.
template <typename T>
std::vector<Vector3<T>> PoissonDiskSampling(T r, const std::array<T, 3>& x_min,
                                            const std::array<T, 3>& x_max);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
