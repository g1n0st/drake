#include "drake/multibody/mpm/particles_to_bgeo.h"

#include <Partio.h>

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
                          const std::vector<int>& ID) {
  DRAKE_DEMAND(q.size() == v.size());
  DRAKE_DEMAND(q.size() == m.size());
  // Create a particle data handle.
  Partio::ParticlesDataMutable* particles = Partio::create();
  Partio::ParticleAttribute position;
  Partio::ParticleAttribute velocity;
  Partio::ParticleAttribute mass;
  Partio::ParticleAttribute id;
  position = particles->addAttribute("position", Partio::VECTOR, 3);
  velocity = particles->addAttribute("velocity", Partio::VECTOR, 3);
  mass = particles->addAttribute("mass", Partio::VECTOR, 1);
  id = particles->addAttribute("id", Partio::INT, 1);
  for (size_t i = 0; i < q.size(); ++i) {
    int index = particles->addParticle();
    // N.B. PARTIO doesn't support double!
    float* q_dest = particles->dataWrite<float>(position, index);
    float* v_dest = particles->dataWrite<float>(velocity, index);
    float* m_dest = particles->dataWrite<float>(mass, index);
    int* id_desk = particles->dataWrite<int>(id, index);
    m_dest[0] = m[i];
    id_desk[0] = ID[i];
    for (int d = 0; d < 3; ++d) {
      q_dest[d] = q[i](d);
      v_dest[d] = v[i](d);
    }
  }
  Partio::write(filename.c_str(), *particles);
  particles->release();
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
