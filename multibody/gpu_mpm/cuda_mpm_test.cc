#include "multibody/gpu_mpm/cuda_mpm_model.cuh"
#include "multibody/gpu_mpm/cuda_mpm_solver.cuh"

#include <gtest/gtest.h>
#include <random>
#include <string>
#include <Partio.h>

namespace drake {
namespace {

using Vec3 = multibody::gmpm::Vec3<double>;

void WriteParticlesToBgeo(const std::string& filename,
                          const std::vector<Vec3>& q,
                          const std::vector<Vec3>& v) {
  // Create a particle data handle.
  Partio::ParticlesDataMutable* particles = Partio::create();
  Partio::ParticleAttribute position;
  Partio::ParticleAttribute velocity;
  position = particles->addAttribute("position", Partio::VECTOR, 3);
  velocity = particles->addAttribute("velocity", Partio::VECTOR, 3);
  for (size_t i = 0; i < q.size(); ++i) {
    int index = particles->addParticle();
    // N.B. PARTIO doesn't support double!
    float* q_dest = particles->dataWrite<float>(position, index);
    float* v_dest = particles->dataWrite<float>(velocity, index);
    for (int d = 0; d < 3; ++d) {
      q_dest[d] = q[i](d);
      v_dest[d] = v[i](d);
    }
  }
  Partio::write(filename.c_str(), *particles);
  particles->release();
}

GTEST_TEST(EstTest, SmokeTest) {
  
  multibody::gmpm::GpuMpmState<double> mpm_state;

  std::vector<multibody::gmpm::Vec3<double>> inital_pos;
  std::vector<multibody::gmpm::Vec3<double>> inital_vel;
  
  
  // Randomly sampling in [0.4-0.6]^3 with 1K particles
  // TODO(changyu): integrate poisson disk sampler
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.45, 0.55);
  for (int i = 0; i < 10000; ++i) {
    inital_pos.emplace_back(dis(gen), dis(gen), dis(gen) - 0.3);
    inital_vel.emplace_back(0, 0, -0.1);
  }

  mpm_state.InitializeParticles(inital_pos, inital_vel, 1000.0);

  EXPECT_TRUE(mpm_state.current_positions() != nullptr);

  multibody::gmpm::GpuMpmSolver<double> mpm_solver;
  double dt = 1e-3;
  for (int frame = 0; frame < 200; frame++) {
    for (int substep = 0; substep < 40; substep++) {
      mpm_solver.RebuildMapping(&mpm_state);
      mpm_solver.ParticleToGrid(&mpm_state, dt);
      mpm_solver.UpdateGrid(&mpm_state);
      mpm_solver.GridToParticle(&mpm_state, dt);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    
    std::vector<multibody::gmpm::Vec3<double>> export_pos;
    std::vector<multibody::gmpm::Vec3<double>> export_vel;
    export_pos.resize(mpm_state.n_particles());
    export_vel.resize(mpm_state.n_particles());
    CUDA_SAFE_CALL(cudaMemcpy(export_pos.data(), mpm_state.current_positions(), sizeof(Vec3) * mpm_state.n_particles(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(export_vel.data(), mpm_state.current_velocities(), sizeof(Vec3) * mpm_state.n_particles(), cudaMemcpyHostToDevice));

    WriteParticlesToBgeo("test" + std::to_string(frame) + ".bgeo", export_pos, export_vel);
  }

  mpm_state.Destroy();
}

}  // namespace
}  // namespace drake