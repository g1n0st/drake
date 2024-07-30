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

using Vec3 = multibody::gmpm::Vec3<float>;


GTEST_TEST(EstTest, SmokeTest) {
  
  multibody::gmpm::GpuMpmState<float> mpm_state;

  std::vector<multibody::gmpm::Vec3<float>> inital_pos;
  std::vector<multibody::gmpm::Vec3<float>> inital_vel;
  std::vector<int> indices;
  
  const int res = 100;
  const float l = 0.5;
  int length = res;
  int width = res;
  float dx = l / width;

  auto p = [&](int i, int j) {
    return i * width + j;
  };

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < width; ++j) {
      inital_pos.emplace_back(0.25f + i * dx, 0.25f + j * dx, 0.75f);
      inital_vel.emplace_back(0.f, 0.f, 0.f);
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

  multibody::gmpm::GpuMpmSolver<float> mpm_solver;
  float dt = 5e-4;
  for (int frame = 0; frame < 240; frame++) {
    long long before_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int substep = 0; substep < 20; substep++) {
      // NOTE, TODO (changyu): DON'T DO sort until we can correctly handle the index mapping.
      mpm_solver.RebuildMapping(&mpm_state, false);
      mpm_solver.ParticleToGrid(&mpm_state, dt);
      mpm_solver.UpdateGrid(&mpm_state);
      mpm_solver.GridToParticle(&mpm_state, dt);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    long long after_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("\033[32mstep=%d time=%lldms\033[0m\n", frame, (after_ts - before_ts));
    
    std::vector<multibody::gmpm::Vec3<float>> export_pos;
    std::vector<multibody::gmpm::Vec3<float>> export_vel;
    export_pos.resize(mpm_state.n_particles());
    export_vel.resize(mpm_state.n_particles());
    CUDA_SAFE_CALL(cudaMemcpy(export_pos.data(), mpm_state.current_positions(), sizeof(Vec3) * mpm_state.n_particles(), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(export_vel.data(), mpm_state.current_velocities(), sizeof(Vec3) * mpm_state.n_particles(), cudaMemcpyDeviceToHost));

    std::ofstream obj("test" + std::to_string(frame) + ".obj");
    for (size_t i = 0; i < mpm_state.n_verts(); ++i) {
      const auto &vert = export_pos[i + mpm_state.n_faces()];
      obj << "v " << vert[0] << " " << vert[1] << " " << vert[2] << "\n";
    }
    for (size_t i = 0; i < mpm_state.n_faces(); ++i) {
      obj << "f " << indices[i*3+0]+1 << " " << indices[i*3+1]+1 << " " << indices[i*3+2]+1 << "\n";
    }
  }

  mpm_state.Destroy();
}

}  // namespace
}  // namespace drake