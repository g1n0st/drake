#include "multibody/gpu_mpm/cuda_mpm_model.cuh"
#include "multibody/gpu_mpm/cuda_mpm_solver.cuh"

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <Partio.h>

namespace drake {
namespace {

using T = float;
using Vec3 = multibody::gmpm::Vec3<T>;


GTEST_TEST(EstTest, SmokeTest) {
  
  multibody::gmpm::GpuMpmState<T> mpm_state;

  std::vector<multibody::gmpm::Vec3<T>> inital_pos;
  std::vector<multibody::gmpm::Vec3<T>> inital_vel;
  std::vector<int> indices;
  
  const int res = 100;
  const T l = T(0.5);
  int length = res;
  int width = res;
  T dx = l / width;

  auto p = [&](int i, int j) {
    return i * width + j;
  };

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < width; ++j) {
      inital_pos.emplace_back(T(0.25 + i * dx), T(0.25 + j * dx), T(0.75));
      inital_vel.emplace_back(T(0.), T(0.), T(0.f));
    }
  }

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < width; ++j) {
      if (i < length - 1 && j < width - 1) {
        indices.push_back(p(i, j));
        indices.push_back(p(i+1, j));
        indices.push_back(p(i, j+1));

        indices.push_back(p(i+1, j+1));
        indices.push_back(p(i, j+1));
        indices.push_back(p(i+1, j));
      }
    }
  }

  mpm_state.InitializeQRCloth(inital_pos, inital_vel, indices);

  EXPECT_TRUE(mpm_state.current_positions() != nullptr);

  multibody::gmpm::GpuMpmSolver<T> mpm_solver;
  T dt = T(1e-3);
  for (int frame = 0; frame < 200; frame++) {
    long long before_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int substep = 0; substep < 40; substep++) {
      mpm_solver.RebuildMapping(&mpm_state, substep == 0);
      mpm_solver.CalcFemStateAndForce(&mpm_state, dt);
      mpm_solver.ParticleToGrid(&mpm_state, dt);
      mpm_solver.UpdateGrid(&mpm_state);
      mpm_solver.GridToParticle(&mpm_state, dt);
    }
    mpm_solver.GpuSync();
    long long after_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("\033[32mstep=%d time=%lldms\033[0m\n", frame, (after_ts - before_ts));
    mpm_solver.Dump(mpm_state, "test" + std::to_string(frame) + ".obj");
  }

  mpm_state.Destroy();
}

}  // namespace
}  // namespace drake