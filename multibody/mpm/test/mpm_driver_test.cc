#include "../mpm_driver.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using math::RigidTransformd;

using Eigen::Vector3d;

GTEST_TEST(MpmDriverTest, Smoke) {
  const double dt = 0.001;
  const double dx = 0.01;
  MpmDriver<double> driver(dt, dx);

  const RigidTransformd X_WB = RigidTransformd::Identity();
  auto sphere_instance = std::make_unique<geometry::GeometryInstance>(
      X_WB, geometry::Sphere(0.05), "sphere");
  const int particles_per_cell = 8;
  const fem::DeformableBodyConfig<double> config;
  driver.SampleParticles(std::move(sphere_instance), particles_per_cell,
                         config);
  EXPECT_NO_THROW(driver.AdvanceOneTimeStep());
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
