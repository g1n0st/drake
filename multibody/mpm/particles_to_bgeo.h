#pragma once

#include <string>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/particles.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Writes a subset of particle information to file.
 @param[in] filename  Absolute path to the file.
 @throws exception if the file with `filename` cannot be written to. */
template <typename T>
void WriteParticlesToBgeo(const std::string& filename,
                          const ParticleData<T>& particle_data);

/* Reads positions, velocities, and mass information from file.
 @param[in] filename  Absolute path to the file.
 @pre q, v, and m are not nullptrs.
 @throws exception if the file with `filename` doesn't exist. */
template <typename T>
void ReadParticlesFromBgeo(const std::string& filename,
                           std::vector<Vector3<T>>* q,
                           std::vector<Vector3<T>>* v, std::vector<T>* m);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
