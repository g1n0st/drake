#pragma once

#include <string>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Writes particle information to file.
 @param[in] filename  Absolute path to the file.
 @param[in] q  Positions of the particles.
 @param[in] v  Velocities of the particles.
 @param[in] m  Masses of the particles.
 @pre q.size() == v.size() == m.size().
 @throws exception if the file with `filename` cannot be written to. */
 void WriteParticlesToBgeo(const std::string& filename,
                          const std::vector<Vector3<double>>& q,
                          const std::vector<Vector3<double>>& v,
                          const std::vector<double>& m,
                          const std::vector<int>& ID);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
