#include "drake/multibody/mpm/poisson_disk.h"

#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kTolerance = 4 * std::numeric_limits<double>::epsilon();

GTEST_TEST(PoissonDiskTest, TestOnePoint) {
  const double r = 2;

  const std::array<double, 3> x_min{-0.1, -0.1, -0.1};
  const std::array<double, 3> x_max{0.1, 0.1, 0.1};

  const std::vector<Vector3<double>> result =
      PoissonDiskSampling(r, x_min, x_max);

  /* Radius is larger than diagonal of bounding box, should return only one
   point. */
  EXPECT_EQ(ssize(result), 1);
  /* The sampled point is in the bounding box. */
  EXPECT_LT(result[0].cwiseAbs().maxCoeff(), 0.1);
}

GTEST_TEST(PoissonDiskTest, TestDistance) {
  const double r = 0.05;

  const std::array<double, 3> x_min{-0.1, -0.1, -0.1};
  const std::array<double, 3> x_max{0.1, 0.1, 0.1};

  const std::vector<Vector3<double>> result =
      PoissonDiskSampling(r, x_min, x_max);

  for (int i = 0; i < ssize(result); ++i) {
    /* Every sampled point is inside the bounding box. */
    EXPECT_LT(result[i].cwiseAbs().maxCoeff(), 0.1);
  }

  /* There's at least two points 0.05 away in a cube with edge length 0.2. */
  EXPECT_TRUE(ssize(result) > 1);

  for (int i = 0; i < ssize(result); ++i) {
    for (int j = i + 1; j < ssize(result); ++j) {
      double distance_sq = (result[i] - result[j]).squaredNorm();
      EXPECT_GT(distance_sq, r * r);
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
