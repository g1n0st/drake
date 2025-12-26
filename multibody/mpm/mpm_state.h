#pragma once

#include <memory>
#include <unordered_set>
#include <vector>
#include <iostream>

#include "drake/geometry/query_results/mpm_particle_contact_pair.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/mpm/internal/analytic_level_set.h"
#include "drake/multibody/mpm/internal/poisson_disk.h"
#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
struct MpmState {
  explicit MpmState(double h) : sparse_grid(h) { DRAKE_DEMAND(h > 0); }

  int AddParticlesUniformly(
      const internal::AnalyticLevelSet& level_set,
      const math::RigidTransform<double>& pose,
      const mpm::constitutive_model::ElastoPlasticModel<T>& elastoplastic_model,
      double common_density, int particles_per_dim) {
    const std::array<Vector3<double>, 2>& bounding_box =
        level_set.bounding_box();

    double max_length = std::max(bounding_box[1](0) - bounding_box[0](0),
                                 bounding_box[1](1) - bounding_box[0](1));
    max_length = std::max(bounding_box[1](2) - bounding_box[0](2), max_length);

    double dx = max_length / (particles_per_dim+1);

    double vol = level_set.volume() /
                 ((particles_per_dim+2) * (particles_per_dim) * (particles_per_dim));
    double mp = vol * common_density;
    for (int i = 0; i <= particles_per_dim+1; ++i) {
      for (int j = 1; j <= particles_per_dim+0; ++j) {
        for (int k = 1; k <= particles_per_dim+0; ++k) {
          particles.AddParticle(
              pose * (bounding_box[0] + Vector3<T>(i * dx, j * dx, k * dx)),
              Vector3<T>(0, 0, 0), elastoplastic_model.Clone(), mp, vol);
        }
      }
    }
    return particles_per_dim * particles_per_dim * particles_per_dim;
  }

  int AddParticlesViaPoissonDiskSampling(
      const internal::AnalyticLevelSet& level_set,
      const math::RigidTransform<double>& pose,
      const mpm::constitutive_model::ElastoPlasticModel<T>& elastoplastic_model,
      double common_density, int min_num_particles_per_cell) {
    const std::array<Vector3<double>, 2> bounding_box =
        level_set.bounding_box();
    // TODO(zeshunzong): pass is input
    double sample_r =
        sparse_grid.h() / (std::cbrt(min_num_particles_per_cell) + 1);

    std::array<double, 3> xmin = {bounding_box[0][0], bounding_box[0][1],
                                  bounding_box[0][2]};
    std::array<double, 3> xmax = {bounding_box[1][0], bounding_box[1][1],
                                  bounding_box[1][2]};
    // Generate sample particles in the reference frame
    std::vector<Vector3<double>> particles_sample_positions =
        internal::PoissonDiskSampling(sample_r, xmin, xmax);

    // Pick out sampled particles that are in the object
    int num_samples = particles_sample_positions.size();
    std::vector<Vector3<double>> particles_positions;
    for (int p = 0; p < num_samples; ++p) {
      // If the particle is in the level set, append its post-transfromed
      // position
      if (level_set.IsInClosure(particles_sample_positions[p])) {
        particles_positions.emplace_back(pose * particles_sample_positions[p]);
      }
    }

    int num_particles = particles_positions.size();
    // We assume every particle have the same volume and mass and constitutive
    // model
    double reference_volume_p = level_set.volume() / num_particles;
    double mass_p = common_density * reference_volume_p;
    // we assume all particles start with zero velocity
    for (int p = 0; p < num_particles; ++p) {
      particles.AddParticle(particles_positions[p], Vector3<T>(0, 0, 0),
                            elastoplastic_model.Clone(), mass_p,
                            reference_volume_p);
    }
    // particles.AddParticle(Vector3<T>(0, 0, 0.0), Vector3<T>(0, 0, 0.0),
    //                       elastoplastic_model.Clone(), mass_p,
    //                       reference_volume_p);
    HackTear();
    return num_particles;  // return the number of particles added
  }

  void HackTear() {
    for (size_t p = 0; p < particles.num_particles(); ++p) {
      if ((particles.GetPositionAt(p)[1] > 0.27) && (particles.GetPositionAt(p)[1] < 0.29)) {
        particles.ScaleYoungsModulus(p, 0.21);
      }
    }
  }

  Particles<T> particles{};
  SparseGrid<T> sparse_grid;
};

template <typename T>
struct TransferScratch {
  // scratch pads for transferring states from particles to grid nodes
  // there will be one pad for each particle
  std::vector<P2gPad<T>> p2g_pads{};
  // scratch pad for transferring states from grid nodes to particles
  G2pPad<T> g2p_pad{};
};

template <typename T>
struct MpmSolverScratch {
  std::vector<Vector3<T>> v_prev;  // previous step grid v
  Eigen::VectorX<T> minus_dEdv;
  Eigen::VectorX<T> dG;  // change to be reflected on grid data
  std::vector<size_t> nodes_collide_with_ground;
  MatrixX<T> d2Edv2;
  TransferScratch<T> transfer_scratch;
  ParticlesData<T> particles_data;

  std::vector<size_t> collision_nodes;
};

template <typename T>
struct MpmGridNodesPermutation {
  std::unordered_set<int> nodes_in_contact;
  std::unordered_set<int> nodes_not_in_contact;

  contact_solvers::internal::PartialPermutation permutation;
  // first nodes in contact, then nodes not in contact

  MpmGridNodesPermutation() {}

  void Compute(const std::vector<geometry::internal::MpmParticleContactPair<T>>&
                   contact_pairs,
               const mpm::MpmState<T>& state) {
    // TODO(zeshunzong): redundancy can be optimized
    // first find all nodes that are in contact
    nodes_in_contact.clear();
    nodes_not_in_contact.clear();
    for (const auto& pair : contact_pairs) {
      const Vector3<int>& base_node =
          state.particles.GetBaseNodeAt(pair.particle_in_contact_index);
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          for (int k = -1; k <= 1; ++k) {
            nodes_in_contact.insert(
                state.sparse_grid.To1DIndex(base_node + Vector3<int>(i, j, k)));
          }
        }
      }
    }

    std::vector<int> new_ordering;
    for (int i = 0; i < static_cast<int>(state.sparse_grid.num_active_nodes());
         ++i) {
      if (nodes_in_contact.count(i)) {
        // i is in contact
        new_ordering.push_back(i);
      }
    }
    for (int i = 0; i < static_cast<int>(state.sparse_grid.num_active_nodes());
         ++i) {
      if (!nodes_in_contact.count(i)) {
        // i is NOT in contact
        new_ordering.push_back(i);
        nodes_not_in_contact.insert(i);
      }
    }
    permutation = contact_solvers::internal::PartialPermutation(new_ordering);
  }
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
