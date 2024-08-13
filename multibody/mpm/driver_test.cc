#include <chrono>
#include <iostream>
#include <string>

#include "mpm_driver.h"

#include "drake/multibody/mpm/particles_to_bgeo.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Vector3d;

int do_main() {
  MpmDriver<float> driver(0.001, 0.01, Parallelism(12));
  math::RigidTransform<double> X_WG(Vector3d(0, 0, 0.25));
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      X_WG, geometry::Sphere(0.10), "sphere");
  fem::DeformableBodyConfig<float> body_config;
  body_config.set_material_model(fem::MaterialModel::kStvkHenckyVonMises);
  body_config.set_youngs_modulus(1e4);
  body_config.set_poissons_ratio(0.3);
  body_config.set_yield_stress(2.5e3);
  driver.SampleParticles(std::move(geometry_instance), 8, body_config);
  const int kNumSteps = 2000;
  const std::string directory = "/home/xuchenhan/Desktop/mpm_data/";
  for (int i = 0; i < kNumSteps; ++i) {
    driver.AdvanceOneTimeStep();
    if (i % 40 == 0) {
      std::cout << "Step " << i << std::endl;
      const std::string filename = fmt::format("particles_{:04d}.bgeo", i/40);
      WriteParticlesToBgeo<float>(directory + filename, driver.particles());
    }
  }
  return 0;
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::multibody::mpm::internal::do_main();
}